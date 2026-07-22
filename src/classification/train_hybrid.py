"""
Approche hybride : CNN (extracteur de features) + SVM (classifieur).

Principe :
  1. Charger EfficientNet-B0 pré-entraîné (ImageNet) — SANS fine-tuning
  2. Retirer la tête de classification → le réseau produit un vecteur 1280-dim
  3. Extraire ces features pour chaque ROI
  4. Entraîner un SVM RBF sur les features CNN

Intérêt scientifique (pour le rapport) :
  - Si CNN+SVM > SVM seul → les features apprises sont plus discriminantes
  - Si CNN+SVM ≈ CNN fine-tuné → le SVM est un bon classifieur
  - Combine interprétabilité (SVM) et richesse des features (CNN)

Usage :
    python -m src.classification.train_hybrid
"""

import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "rois"
OUTPUT = ROOT / "outputs" / "classification" / "hybrid"

QUALITY_LABELS = ["bonne", "moyenne", "mauvaise"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Extracteur de features CNN ───────────────────────────────────────────

def create_feature_extractor():
    """
    EfficientNet-B0 sans la tête : input image → output 1280-dim vector.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Identity()  # retire la tête → output 1280-dim
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def extract_cnn_features(model, split: str, batch_size: int = 64):
    """
    Parcourir toutes les ROI d'un split et extraire les features CNN.

    Returns
    -------
    X : np.ndarray (N, 1280)
    y : np.ndarray (N,)
    """
    all_features = []
    all_labels = []

    for quality_idx, quality_name in enumerate(QUALITY_LABELS):
        quality_dir = ROI_DIR / split / quality_name
        if not quality_dir.exists():
            continue

        images = sorted(quality_dir.glob("*.jpg"))
        batch_tensors = []

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            tensor = eval_transform(img)
            batch_tensors.append(tensor)
            all_labels.append(quality_idx)

            # Processus par batch
            if len(batch_tensors) >= batch_size:
                batch = torch.stack(batch_tensors).to(DEVICE)
                features = model(batch).cpu().numpy()
                all_features.append(features)
                batch_tensors = []

        # Dernier batch partiel
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(DEVICE)
            features = model(batch).cpu().numpy()
            all_features.append(features)

    if not all_features:
        return np.array([]), np.array([])

    X = np.vstack(all_features)
    y = np.array(all_labels)
    return X, y


# ── Entraînement ─────────────────────────────────────────────────────────

def train():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Hybride — CNN Features (EfficientNet-B0) + SVM")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # ── 1. Extraction des features CNN ────────────────────────────────
    print("\n[1/4] Création de l'extracteur CNN (pré-entraîné ImageNet)...")
    extractor = create_feature_extractor()

    print("\n[INFO] Extraction des features CNN par split...")

    print("  → train3...")
    X_train, y_train = extract_cnn_features(extractor, "train3")
    print(f"    {len(X_train)} ROI → vecteurs {X_train.shape[1]}-dim")

    print("  → val3...")
    X_val, y_val = extract_cnn_features(extractor, "val3")
    print(f"    {len(X_val)} ROI")

    print("  → test3...")
    X_test, y_test = extract_cnn_features(extractor, "test3")
    print(f"    {len(X_test)} ROI")

    if len(X_train) == 0:
        raise RuntimeError(
            "Aucune ROI trouvée. Exécutez d'abord :\n"
            "  python -m src.classification.extract_rois"
        )

    # Libérer la mémoire GPU
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. GridSearchCV sur SVM ───────────────────────────────────────
    print("\n[2/4] GridSearchCV (SVM RBF sur features CNN)...")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=1, refit=True,
    )
    grid.fit(X_trainval, y_trainval)

    print(f"\n  Meilleurs params : {grid.best_params_}")
    print(f"  F1-macro (CV)    : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # ── 3. Évaluation sur test3 ───────────────────────────────────────
    print("\n[3/4] Évaluation sur test3...")

    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test, y_pred, target_names=QUALITY_LABELS, digits=4,
    )
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(report)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1:.4f}")

    # Sauvegarder rapport
    with open(OUTPUT / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Approche : Hybride CNN (EfficientNet-B0) + SVM RBF\n")
        f.write(f"Features : 1280-dim (couche pré-classification)\n")
        f.write(f"Params SVM : {grid.best_params_}\n")
        f.write(f"F1-macro (CV) : {grid.best_score_:.4f}\n\n")
        f.write(f"--- Résultats sur test3 ---\n\n")
        f.write(report)
        f.write(f"\nAccuracy : {acc:.4f}\n")
        f.write(f"F1-macro : {f1:.4f}\n")

    with open(OUTPUT / "results.json", "w") as f:
        json.dump({
            "approach": "hybrid_cnn_svm",
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "best_params": grid.best_params_,
            "cv_f1": round(grid.best_score_, 4),
        }, f, indent=2)

    # ── 4. Matrice de confusion ───────────────────────────────────────
    print("\n[4/4] Matrice de confusion...")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=QUALITY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_title("Hybride (CNN+SVM) — Matrice de confusion (test3)", fontsize=13)
    fig.savefig(OUTPUT / "confusion_matrix_hybrid.png", dpi=150, bbox_inches="tight")
    plt.close()

    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    disp_n = ConfusionMatrixDisplay(cm_norm, display_labels=QUALITY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_n.plot(ax=ax, cmap="Greens", values_format=".2%")
    ax.set_title("Hybride — Matrice normalisée (test3)", fontsize=13)
    fig.savefig(OUTPUT / "confusion_matrix_hybrid_norm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Sauvegarde modèle
    with open(OUTPUT / "hybrid_svm_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\n  Résultats sauvegardés dans : {OUTPUT}")

    print("\n" + "=" * 60)
    print("  [DONE] Approche hybride terminée")
    print("=" * 60)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1:.4f}")
    print(f"\n  Prochaine étape : python -m src.classification.compare")

    return best_model, acc, f1


if __name__ == "__main__":
    train()
