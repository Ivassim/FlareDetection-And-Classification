"""
Entraînement du classifieur SVM sur les features visuelles manuelles.

Pipeline :
  1. Charger les ROI extraites (data/rois/)
  2. Extraire les 113 features pour chaque ROI
  3. GridSearchCV (5-fold stratifié) pour optimiser C et gamma
  4. Évaluer sur test3 : matrice confusion, P/R/F1, rapport
  5. Sauvegarder modèle + résultats

Usage :
    python -m src.classification.train_svm
"""

import pickle
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
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

from src.classification.features import extract_all_features

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "rois"
OUTPUT = ROOT / "outputs" / "classification" / "svm"

QUALITY_LABELS = ["bonne", "moyenne", "mauvaise"]
LABEL_TO_INT = {q: i for i, q in enumerate(QUALITY_LABELS)}


# ── Chargement des données ───────────────────────────────────────────────

def load_rois(split: str, verbose: bool = True):
    """
    Charge toutes les ROI d'un split et extrait les features.

    Returns
    -------
    X : np.ndarray (N, 113)
    y : np.ndarray (N,)
    """
    X, y = [], []
    split_dir = ROI_DIR / split

    for quality in QUALITY_LABELS:
        quality_dir = split_dir / quality
        if not quality_dir.exists():
            if verbose:
                print(f"  [WARN] Dossier manquant : {quality_dir}")
            continue

        images = sorted(quality_dir.glob("*.jpg"))
        count = 0
        for img_path in images:
            roi = cv2.imread(str(img_path))
            if roi is None or roi.size == 0:
                continue
            try:
                features = extract_all_features(roi)
                X.append(features)
                y.append(LABEL_TO_INT[quality])
                count += 1
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Erreur extraction {img_path.name}: {e}")

        if verbose:
            print(f"  {split}/{quality:<12}: {count:>5} ROI chargées")

    return np.array(X), np.array(y)


# ── Entraînement ─────────────────────────────────────────────────────────

def train():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SVM — Classification qualité de combustion")
    print("=" * 60)

    # ── 1. Chargement ─────────────────────────────────────────────────
    print("\n[1/5] Chargement et extraction des features...\n")
    X_train, y_train = load_rois("train3")
    X_val, y_val = load_rois("val3")
    X_test, y_test = load_rois("test3")

    print(f"\n  Résumé :")
    print(f"    Train : {len(X_train):>5} ROI  ({X_train.shape[1]} features)")
    print(f"    Val   : {len(X_val):>5} ROI")
    print(f"    Test  : {len(X_test):>5} ROI")

    if len(X_train) == 0:
        raise RuntimeError(
            "Aucune ROI trouvée. Exécutez d'abord :\n"
            "  python -m src.classification.extract_rois"
        )

    # ── 2. GridSearchCV ───────────────────────────────────────────────
    print("\n[2/5] GridSearchCV (C, gamma) sur train+val...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.001],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Combiner train + val pour la recherche (test reste indépendant)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_trainval, y_trainval)

    print(f"\n  Meilleurs hyperparamètres : {grid.best_params_}")
    print(f"  Meilleur F1-macro (CV)    : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # ── 3. Évaluation sur test3 ───────────────────────────────────────
    print("\n[3/5] Évaluation sur test3 (jamais vu)...\n")

    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=QUALITY_LABELS, digits=4,
    )
    print(report)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1-macro    : {f1:.4f}")

    # Sauvegarder le rapport texte
    report_path = OUTPUT / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Approche : SVM RBF sur features manuelles (113 dim)\n")
        f.write(f"Meilleurs hyperparamètres : {grid.best_params_}\n")
        f.write(f"F1-macro (CV 5-fold)      : {grid.best_score_:.4f}\n")
        f.write(f"\n--- Résultats sur test3 ---\n\n")
        f.write(report)
        f.write(f"\nAccuracy : {acc:.4f}\n")
        f.write(f"F1-macro : {f1:.4f}\n")
    print(f"  Rapport sauvegardé : {report_path}")

    # ── 4. Matrice de confusion ───────────────────────────────────────
    print("\n[4/5] Matrices de confusion...")

    # Absolue
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=QUALITY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("SVM — Matrice de confusion (test3)", fontsize=14)
    fig.savefig(OUTPUT / "confusion_matrix_svm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Normalisée
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    disp_n = ConfusionMatrixDisplay(cm_norm, display_labels=QUALITY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_n.plot(ax=ax, cmap="Blues", values_format=".2%")
    ax.set_title("SVM — Matrice de confusion normalisée (test3)", fontsize=14)
    fig.savefig(
        OUTPUT / "confusion_matrix_svm_normalized.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"  Matrices sauvegardées dans : {OUTPUT}")

    # ── 5. Sauvegarde du modèle ───────────────────────────────────────
    print("\n[5/5] Sauvegarde du modèle...")

    model_path = OUTPUT / "svm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Modèle sauvegardé : {model_path}")

    # Résumé final
    print("\n" + "=" * 60)
    print("  [DONE] Entraînement SVM terminé")
    print("=" * 60)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1-macro    : {f1:.4f}")
    print(f"  Résultats   : {OUTPUT}")
    print(f"\n  Prochaine étape : python -m src.classification.ablation_svm")

    return best_model, acc, f1


if __name__ == "__main__":
    train()
