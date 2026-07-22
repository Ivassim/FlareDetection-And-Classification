"""
Fine-tuning EfficientNet-B0 pour classification qualité de combustion.

Architecture :
  EfficientNet-B0 (pré-entraîné ImageNet, ~5.3M params)
    → Average Pool → 1280-dim
    → Dropout(0.3)
    → Linear(1280, 3)   classes : [bonne, moyenne, mauvaise]

Stratégie de fine-tuning en 2 phases :
  Phase 1 (epochs 1→10)  : backbone gelé, seule la tête s'entraîne
  Phase 2 (epochs 11→30) : 2 derniers blocs dégelés, LR réduit ×10

Gestion du déséquilibre :
  WeightedRandomSampler surpondere les classes rares dans le DataLoader.

Usage :
    python -m src.classification.train_cnn
"""

import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "rois"
OUTPUT = ROOT / "outputs" / "classification" / "cnn"

QUALITY_LABELS = ["bonne", "moyenne", "mauvaise"]
NUM_CLASSES = len(QUALITY_LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparamètres ──────────────────────────────────────────────────────

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
LABEL_SMOOTHING = 0.05


# ── Dataset ──────────────────────────────────────────────────────────────

class FlareROIDataset(Dataset):
    """Dataset PyTorch pour les ROI de flammes."""

    def __init__(self, split: str, transform=None):
        self.samples = []
        self.transform = transform

        split_dir = ROI_DIR / split
        for quality_idx, quality_name in enumerate(QUALITY_LABELS):
            quality_dir = split_dir / quality_name
            if not quality_dir.exists():
                continue
            for img_path in sorted(quality_dir.glob("*.jpg")):
                self.samples.append((img_path, quality_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# Transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Modèle ───────────────────────────────────────────────────────────────

def create_model():
    """Crée un EfficientNet-B0 avec tête de classification personnalisée."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def freeze_backbone(model):
    """Geler le backbone (features) — seule la tête est entraînable."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model, n_blocks=2):
    """Dégeler les N derniers blocs du backbone pour fine-tuning."""
    total_blocks = len(model.features)
    for i in range(total_blocks - n_blocks, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True


# ── Boucles d'entraînement / évaluation ──────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Un epoch de training."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """Évaluation sur un split (val ou test)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Entraînement principal ───────────────────────────────────────────────

def train():
    # Reproductibilité
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  CNN — EfficientNet-B0 Fine-tuning")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # ── 1. Datasets ───────────────────────────────────────────────────
    print("\n[1/6] Chargement des datasets...")
    train_ds = FlareROIDataset("train3", train_transform)
    val_ds = FlareROIDataset("val3", eval_transform)
    test_ds = FlareROIDataset("test3", eval_transform)

    print(f"  Train : {len(train_ds):>5} ROI")
    print(f"  Val   : {len(val_ds):>5} ROI")
    print(f"  Test  : {len(test_ds):>5} ROI")

    if len(train_ds) == 0:
        raise RuntimeError(
            "Aucune ROI trouvée. Exécutez d'abord :\n"
            "  python -m src.classification.extract_rois"
        )

    # Weighted sampler pour gérer le déséquilibre
    labels = [s[1] for s in train_ds.samples]
    class_counts = Counter(labels)
    print(f"\n  Distribution train :")
    for idx, name in enumerate(QUALITY_LABELS):
        print(f"    {name:<12}: {class_counts.get(idx, 0):>5}")

    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── 2. Modèle ────────────────────────────────────────────────────
    print("\n[2/6] Création du modèle EfficientNet-B0...")
    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {n_params_total:,}")

    # ── 3. Phase 1 : tête seule ──────────────────────────────────────
    print(f"\n[3/6] Phase 1 — Entraînement de la tête ({EPOCHS_PHASE1} epochs)...")
    freeze_backbone(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {n_trainable:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:02d}/{EPOCHS_PHASE1} | "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | LR={lr:.6f}"
        )
        history.append({
            "epoch": epoch, "phase": 1,
            "tr_loss": tr_loss, "tr_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

    # ── 4. Phase 2 : fine-tune partiel ────────────────────────────────
    print(f"\n[4/6] Phase 2 — Fine-tuning partiel ({EPOCHS_PHASE2} epochs)...")
    unfreeze_last_blocks(model, n_blocks=2)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {n_trainable:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE2, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)
    patience_counter = 0

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step()

        global_epoch = EPOCHS_PHASE1 + epoch
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {global_epoch:02d}/{EPOCHS_PHASE1 + EPOCHS_PHASE2} | "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | LR={lr:.6f}"
        )
        history.append({
            "epoch": global_epoch, "phase": 2,
            "tr_loss": tr_loss, "tr_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping à epoch {global_epoch}")
                break

    # ── 5. Évaluation finale ──────────────────────────────────────────
    print("\n[5/6] Évaluation finale sur test3...")

    model.load_state_dict(torch.load(OUTPUT / "best_model.pt", weights_only=True))
    _, test_acc, y_pred, y_true = evaluate_model(
        model, test_loader, criterion, DEVICE,
    )

    report = classification_report(
        y_true, y_pred, target_names=QUALITY_LABELS, digits=4,
    )
    f1 = f1_score(y_true, y_pred, average="macro")

    print(report)
    print(f"  Accuracy    : {test_acc:.4f}")
    print(f"  F1-macro    : {f1:.4f}")
    print(f"  Best val acc: {best_val_acc:.4f}")

    # Rapport texte
    report_path = OUTPUT / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Approche : CNN EfficientNet-B0 (fine-tuning 2 phases)\n")
        f.write(f"Device   : {DEVICE}\n")
        f.write(f"Params   : {n_params_total:,}\n")
        f.write(f"Epochs   : Phase1={EPOCHS_PHASE1} + Phase2={EPOCHS_PHASE2}\n")
        f.write(f"Best val : {best_val_acc:.4f}\n\n")
        f.write(f"--- Résultats sur test3 ---\n\n")
        f.write(report)
        f.write(f"\nAccuracy : {test_acc:.4f}\n")
        f.write(f"F1-macro : {f1:.4f}\n")

    # Rapport JSON
    json_result = {
        "approach": "cnn_efficientnet_b0",
        "accuracy": round(test_acc, 4),
        "f1_macro": round(f1, 4),
        "best_val_acc": round(best_val_acc, 4),
        "epochs_trained": len(history),
        "device": str(DEVICE),
    }
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(json_result, f, indent=2)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=QUALITY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Oranges", values_format="d")
    ax.set_title("CNN (EfficientNet-B0) — Matrice de confusion (test3)", fontsize=13)
    fig.savefig(OUTPUT / "confusion_matrix_cnn.png", dpi=150, bbox_inches="tight")
    plt.close()

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    disp_n = ConfusionMatrixDisplay(cm_norm, display_labels=QUALITY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_n.plot(ax=ax, cmap="Oranges", values_format=".2%")
    ax.set_title("CNN — Matrice de confusion normalisée (test3)", fontsize=13)
    fig.savefig(OUTPUT / "confusion_matrix_cnn_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Courbes d'entraînement ─────────────────────────────────────
    print("\n[6/6] Génération des courbes d'entraînement...")

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, [h["tr_loss"] for h in history], label="Train", color="#e74c3c")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val", color="#3498db")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss vs Epoch")
    axes[0].axvline(x=EPOCHS_PHASE1, color="gray", linestyle="--", alpha=0.5,
                    label="Phase 1→2")

    # Accuracy
    axes[1].plot(epochs, [h["tr_acc"] for h in history], label="Train", color="#e74c3c")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="Val", color="#3498db")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].axvline(x=EPOCHS_PHASE1, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("EfficientNet-B0 — Courbes d'entraînement", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Sauvegarder l'historique
    with open(OUTPUT / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Résultats sauvegardés dans : {OUTPUT}")

    print("\n" + "=" * 60)
    print("  [DONE] Entraînement CNN terminé")
    print("=" * 60)
    print(f"  Accuracy    : {test_acc:.4f}")
    print(f"  F1-macro    : {f1:.4f}")
    print(f"\n  Prochaine étape : python -m src.classification.train_hybrid")

    return model, test_acc, f1


if __name__ == "__main__":
    train()
