"""
Visualisation des prédictions de classification qualité de combustion.

Génère des grilles d'images montrant :
  1. Grille "Ground Truth" — ROI colorées par label réel
  2. Grille "Prédictions SVM" — ROI colorées par prédiction + indicateur ✓/✗
  3. Grille "Prédictions CNN" — idem pour le CNN
  4. Grille "Erreurs" — uniquement les ROI mal classifiées
  5. Grille comparative par classe — échantillons de chaque qualité

Similaire aux val_batch_labels de YOLO Ultralytics.

Usage :
    python -m src.classification.visualize_predictions
"""

import pickle
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.classification.features import extract_all_features

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "rois"
OUTPUT = ROOT / "outputs" / "classification" / "visualizations"

QUALITY_LABELS = ["bonne", "moyenne", "mauvaise"]
LABEL_TO_INT = {q: i for i, q in enumerate(QUALITY_LABELS)}

# Couleurs par classe (BGR pour OpenCV, RGB pour matplotlib)
COLORS_RGB = {
    "bonne":    (46, 204, 113),   # vert
    "moyenne":  (241, 196, 15),   # jaune/orange
    "mauvaise": (231, 76, 60),    # rouge
}
COLORS_HEX = {
    "bonne":    "#2ecc71",
    "moyenne":  "#f1c40f",
    "mauvaise": "#e74c3c",
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Chargement des ROI ───────────────────────────────────────────────────

def load_test_rois(max_per_class=None):
    """
    Charge les chemins et labels des ROI de test3.
    
    Returns
    -------
    samples : list of (Path, str, int)
        (chemin_image, nom_qualité, index_qualité)
    """
    samples = []
    split_dir = ROI_DIR / "test3"
    
    for quality in QUALITY_LABELS:
        quality_dir = split_dir / quality
        if not quality_dir.exists():
            continue
        paths = sorted(quality_dir.glob("*.jpg"))
        if max_per_class and len(paths) > max_per_class:
            paths = random.sample(paths, max_per_class)
        for p in paths:
            samples.append((p, quality, LABEL_TO_INT[quality]))
    
    random.shuffle(samples)
    return samples


# ── Prédictions SVM ──────────────────────────────────────────────────────

def predict_svm(samples):
    """Prédire avec le modèle SVM sauvegardé."""
    model_path = ROOT / "outputs" / "classification" / "svm" / "svm_model.pkl"
    if not model_path.exists():
        print("  [WARN] Modèle SVM non trouvé, skip")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    predictions = []
    for img_path, true_label, true_idx in samples:
        roi = cv2.imread(str(img_path))
        if roi is None:
            predictions.append(true_idx)
            continue
        features = extract_all_features(roi).reshape(1, -1)
        pred = model.predict(features)[0]
        predictions.append(pred)
    
    return predictions


# ── Prédictions CNN ──────────────────────────────────────────────────────

def predict_cnn(samples):
    """Prédire avec le modèle CNN sauvegardé."""
    model_path = ROOT / "outputs" / "classification" / "cnn" / "best_model.pt"
    if not model_path.exists():
        print("  [WARN] Modèle CNN non trouvé, skip")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 3),
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    predictions = []
    with torch.no_grad():
        for img_path, true_label, true_idx in samples:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)
            pred = output.argmax(1).item()
            predictions.append(pred)
    
    return predictions


# ── Grille de visualisation ──────────────────────────────────────────────

def draw_roi_grid(samples, predictions, title, save_path, 
                  n_cols=8, n_rows=6, cell_size=100):
    """
    Dessine une grille de ROI avec bordures colorées selon la prédiction.
    
    ✓ vert  = prédiction correcte
    ✗ rouge = prédiction incorrecte
    """
    n_total = min(n_cols * n_rows, len(samples))
    if n_total == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.9))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    
    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[col]
        
        if idx < n_total:
            img_path, true_label, true_idx = samples[idx]
            pred_idx = predictions[idx] if predictions else true_idx
            pred_label = QUALITY_LABELS[pred_idx]
            
            # Charger et redimensionner
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (cell_size, cell_size))
            else:
                img = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
            
            ax.imshow(img)
            
            correct = (pred_idx == true_idx)
            
            if predictions is not None:
                # Bordure colorée selon correctness
                border_color = COLORS_HEX[pred_label]
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3 if correct else 4)
                
                if not correct:
                    # Bordure rouge épaisse pour les erreurs
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#e74c3c")
                        spine.set_linewidth(4)
                    # Afficher le vrai label en petit
                    ax.set_xlabel(f"vrai: {true_label}", fontsize=6, color="#e74c3c")
                
                symbol = "✓" if correct else "✗"
                color = "#2ecc71" if correct else "#e74c3c"
                ax.set_title(f"{symbol} {pred_label}", fontsize=7, 
                           color=color, fontweight="bold", pad=2)
            else:
                # Mode ground truth : bordure = couleur de la classe
                border_color = COLORS_HEX[true_label]
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                ax.set_title(true_label, fontsize=7, 
                           color=border_color, fontweight="bold", pad=2)
        else:
            ax.axis("off")
            continue
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Légende
    legend_patches = [
        mpatches.Patch(color=COLORS_HEX[q], label=q.capitalize())
        for q in QUALITY_LABELS
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, 
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {save_path.name}")


def draw_class_samples_grid(save_path, n_per_class=12, cell_size=100):
    """
    Grille comparative : N exemples de chaque classe côte à côte.
    3 colonnes (bonne | moyenne | mauvaise), N lignes.
    """
    fig, axes = plt.subplots(n_per_class, 3, figsize=(6, n_per_class * 1.4))
    fig.suptitle("Exemples de ROI par classe de qualité de combustion", 
                 fontsize=14, fontweight="bold", y=1.01)
    
    for col_idx, quality in enumerate(QUALITY_LABELS):
        quality_dir = ROI_DIR / "test3" / quality
        if not quality_dir.exists():
            continue
        
        images = sorted(quality_dir.glob("*.jpg"))
        selected = random.sample(images, min(n_per_class, len(images)))
        
        # Titre de colonne
        axes[0][col_idx].set_title(
            quality.upper(), fontsize=11, fontweight="bold",
            color=COLORS_HEX[quality], pad=8
        )
        
        for row_idx in range(n_per_class):
            ax = axes[row_idx][col_idx]
            
            if row_idx < len(selected):
                img = cv2.imread(str(selected[row_idx]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (cell_size, cell_size))
                    ax.imshow(img)
                
                for spine in ax.spines.values():
                    spine.set_edgecolor(COLORS_HEX[quality])
                    spine.set_linewidth(2)
            else:
                ax.axis("off")
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {save_path.name}")


def draw_errors_grid(samples, predictions, model_name, save_path,
                     n_cols=6, n_rows=4, cell_size=120):
    """
    Grille des erreurs uniquement.
    Chaque cellule montre : l'image + "prédit: X → vrai: Y"
    """
    errors = []
    for i, (img_path, true_label, true_idx) in enumerate(samples):
        pred_idx = predictions[i]
        if pred_idx != true_idx:
            errors.append((img_path, true_label, true_idx, 
                          QUALITY_LABELS[pred_idx], pred_idx))
    
    if not errors:
        print(f"  [INFO] Aucune erreur pour {model_name} !")
        return
    
    n_show = min(n_cols * n_rows, len(errors))
    show_errors = errors[:n_show]
    actual_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(actual_rows, n_cols, 
                             figsize=(n_cols * 2, actual_rows * 2.3))
    if actual_rows == 1:
        axes = [axes]
    
    fig.suptitle(f"{model_name} — Erreurs de classification ({len(errors)} total)", 
                 fontsize=14, fontweight="bold", color="#e74c3c", y=1.01)
    
    for idx in range(actual_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col] if actual_rows > 1 else axes[0][col]
        
        if idx < n_show:
            img_path, true_label, true_idx, pred_label, pred_idx = show_errors[idx]
            
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (cell_size, cell_size))
                ax.imshow(img)
            
            for spine in ax.spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(3)
            
            ax.set_title(f"prédit: {pred_label}", fontsize=7, 
                        color=COLORS_HEX[pred_label], fontweight="bold", pad=2)
            ax.set_xlabel(f"vrai: {true_label}", fontsize=7, 
                         color=COLORS_HEX[true_label], fontweight="bold")
        else:
            ax.axis("off")
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {save_path.name}  ({len(errors)} erreurs)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  Visualisation des prédictions de classification")
    print("=" * 60)
    
    # Charger les ROI de test
    print("\n[1/6] Chargement des ROI test3...")
    samples = load_test_rois()
    print(f"  {len(samples)} ROI chargées")
    
    # Sous-ensemble pour les grilles (48 images = 6×8)
    grid_samples = samples[:48]
    
    # ── Grille Ground Truth ───────────────────────────────────────────
    print("\n[2/6] Grille Ground Truth...")
    draw_roi_grid(
        grid_samples, predictions=None,
        title="Ground Truth — Labels réels (test3)",
        save_path=OUTPUT / "grid_ground_truth.png",
    )
    
    # ── Grille d'exemples par classe ──────────────────────────────────
    print("\n[3/6] Exemples par classe...")
    draw_class_samples_grid(
        save_path=OUTPUT / "grid_class_samples.png",
        n_per_class=10,
    )
    
    # ── Prédictions SVM ───────────────────────────────────────────────
    print("\n[4/6] Prédictions SVM...")
    svm_preds = predict_svm(grid_samples)
    if svm_preds is not None:
        draw_roi_grid(
            grid_samples, svm_preds,
            title="SVM — Prédictions sur test3 (✓ correct / ✗ erreur)",
            save_path=OUTPUT / "grid_svm_predictions.png",
        )
        # Erreurs SVM sur TOUT le test set
        all_svm_preds = predict_svm(samples)
        draw_errors_grid(
            samples, all_svm_preds, "SVM",
            save_path=OUTPUT / "grid_svm_errors.png",
        )
    
    # ── Prédictions CNN ───────────────────────────────────────────────
    print("\n[5/6] Prédictions CNN...")
    cnn_preds = predict_cnn(grid_samples)
    if cnn_preds is not None:
        draw_roi_grid(
            grid_samples, cnn_preds,
            title="CNN (EfficientNet-B0) — Prédictions sur test3",
            save_path=OUTPUT / "grid_cnn_predictions.png",
        )
        # Erreurs CNN
        all_cnn_preds = predict_cnn(samples)
        draw_errors_grid(
            samples, all_cnn_preds, "CNN",
            save_path=OUTPUT / "grid_cnn_errors.png",
        )
    
    # ── Résumé ────────────────────────────────────────────────────────
    print("\n[6/6] Statistiques...")
    if svm_preds is not None:
        correct_svm = sum(1 for s, p in zip(grid_samples, svm_preds) if s[2] == p)
        print(f"  SVM grid : {correct_svm}/{len(grid_samples)} correct")
    if cnn_preds is not None:
        correct_cnn = sum(1 for s, p in zip(grid_samples, cnn_preds) if s[2] == p)
        print(f"  CNN grid : {correct_cnn}/{len(grid_samples)} correct")
    
    print(f"\n[DONE] Visualisations sauvegardées dans : {OUTPUT}")
    print("\nFichiers générés :")
    for f in sorted(OUTPUT.glob("*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
