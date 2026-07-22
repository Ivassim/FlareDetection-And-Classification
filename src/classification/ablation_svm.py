"""
Ablation study — mesurer la contribution de chaque groupe de features.

Principe :
  Pour chaque groupe (HSV, LBP, GLCM, etc.) :
    1. Retirer ce groupe du vecteur de features
    2. Ré-entraîner le SVM avec les mêmes hyperparamètres
    3. Mesurer le delta d'accuracy et F1

  Si retirer un groupe fait baisser le F1 → ce groupe est important.
  Si retirer un groupe ne change rien → ce groupe est redondant.

Résultat : tableau prêt pour le mémoire.

Usage :
    python -m src.classification.ablation_svm
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.classification.features import extract_all_features, FEATURE_GROUPS

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "rois"
OUTPUT = ROOT / "outputs" / "classification" / "svm"

QUALITY_LABELS = ["bonne", "moyenne", "mauvaise"]
LABEL_TO_INT = {q: i for i, q in enumerate(QUALITY_LABELS)}


def load_rois_features(split: str):
    """Charge les ROI et extrait les features."""
    X, y = [], []
    split_dir = ROI_DIR / split
    for quality in QUALITY_LABELS:
        quality_dir = split_dir / quality
        if not quality_dir.exists():
            continue
        for img_path in sorted(quality_dir.glob("*.jpg")):
            roi = cv2.imread(str(img_path))
            if roi is None or roi.size == 0:
                continue
            try:
                X.append(extract_all_features(roi))
                y.append(LABEL_TO_INT[quality])
            except Exception:
                continue
    return np.array(X), np.array(y)


def run_ablation():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Ablation Study — Contribution des features")
    print("=" * 65)

    # Charger les données
    print("\n[INFO] Chargement des features...")
    X_train, y_train = load_rois_features("train3")
    X_val, y_val = load_rois_features("val3")
    X_test, y_test = load_rois_features("test3")

    # Combiner train + val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Charger les meilleurs hyperparamètres depuis le modèle entraîné
    model_path = OUTPUT / "svm_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        C = saved.named_steps["svm"].C
        gamma = saved.named_steps["svm"].gamma
        print(f"  Hyperparamètres du modèle sauvegardé : C={C}, gamma={gamma}")
    else:
        C, gamma = 10, "scale"
        print(f"  Modèle non trouvé, utilisation des défauts : C={C}, gamma={gamma}")

    # ── Baseline : toutes les features ────────────────────────────────
    pipe_full = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")),
    ])
    pipe_full.fit(X_trainval, y_trainval)
    y_pred_full = pipe_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    f1_full = f1_score(y_test, y_pred_full, average="macro")

    # ── Ablation : retirer un groupe à la fois ────────────────────────
    print(f"\n{'Groupe retiré':<20} | {'Accuracy':>10} | {'F1-macro':>10} | {'Δ F1':>10}")
    print("-" * 65)
    print(f"{'AUCUN (baseline)':<20} | {acc_full:>10.4f} | {f1_full:>10.4f} | {'—':>10}")
    print("-" * 65)

    results = []
    for group_name, (start, end) in FEATURE_GROUPS.items():
        # Créer un masque excluant ce groupe
        mask = np.ones(X_trainval.shape[1], dtype=bool)
        mask[start:end] = False

        X_tr_masked = X_trainval[:, mask]
        X_te_masked = X_test[:, mask]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")),
        ])
        pipe.fit(X_tr_masked, y_trainval)
        y_pred = pipe.predict(X_te_masked)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        delta = f1 - f1_full

        sign = "+" if delta > 0 else ""
        print(f"sans {group_name:<16} | {acc:>10.4f} | {f1:>10.4f} | {sign}{delta:>9.4f}")

        results.append({
            "group": group_name,
            "dims": end - start,
            "accuracy": acc,
            "f1_macro": f1,
            "delta_f1": delta,
        })

    # ── Graphique ─────────────────────────────────────────────────────
    print("\n[INFO] Génération du graphique d'ablation...")

    groups = [r["group"] for r in results]
    deltas = [r["delta_f1"] for r in results]
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(groups, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Δ F1-macro (vs baseline)")
    ax.set_title("Ablation Study — Impact de chaque groupe de features", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Annoter les barres
    for bar, d in zip(bars, deltas):
        sign = "+" if d > 0 else ""
        ax.text(
            bar.get_width() + 0.002 * (1 if d >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{d:.4f}",
            va="center", fontsize=9,
        )

    ax.invert_yaxis()
    fig.tight_layout()
    fig_path = OUTPUT / "ablation_study.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Sauvegarder en texte
    txt_path = OUTPUT / "ablation_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Baseline : Accuracy={acc_full:.4f}  F1-macro={f1_full:.4f}\n\n")
        f.write(f"{'Groupe retiré':<20} {'Dims':>5} {'Acc':>10} {'F1':>10} {'Δ F1':>10}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            sign = "+" if r["delta_f1"] > 0 else ""
            f.write(
                f"sans {r['group']:<16} {r['dims']:>5} "
                f"{r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
                f"{sign}{r['delta_f1']:>9.4f}\n"
            )

    print(f"\n  Graphique : {fig_path}")
    print(f"  Résultats : {txt_path}")
    print("\n[DONE] Ablation study terminée.")

    return results


if __name__ == "__main__":
    run_ablation()
