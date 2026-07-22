# -*- coding: utf-8 -*-
"""
Reentrainement SVM avec un decoupage GROUPE PAR FRAME SOURCE (sans fuite).

Objectif : mesurer l'impact de la fuite de donnees du split d'origine
(train3/val3/test3 melangeaient augmentations/crops d'une meme frame entre
splits). Ici, toutes les ROI d'une meme frame source vont dans le MEME split,
et la validation croisee respecte aussi les groupes (StratifiedGroupKFold).

Config SVM identique a train_svm.py (scaler + SVC RBF class_weight=balanced,
meme grille C/gamma, scoring f1_macro). Rien n'est ecrase : sorties separees.

Usage : python -m src.classification.retrain_svm_grouped
"""
import re
import json
from pathlib import Path

import cv2
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     GroupShuffleSplit)
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.classification.features import extract_all_features

ROOT    = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "rois"
OUTPUT  = ROOT / "outputs" / "classification" / "svm_grouped"
QUALITY = ["bonne", "moyenne", "mauvaise"]
LBL     = {q: i for i, q in enumerate(QUALITY)}


def frame_key(fname: str) -> str:
    """Identifiant de la FRAME SOURCE (on retire augmentations / variantes Roboflow)."""
    return re.split(r"_aug_|_original|_jpg\.rf\.|\.rf\.", fname)[0]


def video_key(fkey: str) -> str:
    return fkey.split("_frame_")[0] if "_frame_" in fkey else fkey


def charger_tout():
    """Charge TOUTES les ROI (train3+val3+test3 confondus) + features + groupe."""
    X, y, groups, vids = [], [], [], []
    n_files = 0
    for split in ("train3", "val3", "test3"):
        for q in QUALITY:
            d = ROI_DIR / split / q
            if not d.exists():
                continue
            for img in sorted(d.glob("*.jpg")):
                roi = cv2.imread(str(img))
                if roi is None or roi.size == 0:
                    continue
                try:
                    feats = extract_all_features(roi)
                except Exception:
                    continue
                X.append(feats); y.append(LBL[q])
                fk = frame_key(img.name)
                groups.append(fk); vids.append(video_key(fk))
                n_files += 1
    return (np.array(X), np.array(y), np.array(groups),
            np.array(vids), n_files)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("  SVM — re-split GROUPE PAR FRAME (sans fuite)")
    print("=" * 64)

    print("\n[1/4] Chargement + extraction des features (toutes les ROI)...")
    X, y, g, vids, n = charger_tout()
    print(f"  ROI chargees        : {n}")
    print(f"  Frames sources uniq.: {len(set(g))}")
    print(f"  Videos sources uniq.: {len(set(vids))}")
    print(f"  Distribution classes: " +
          ", ".join(f"{QUALITY[k]}={int((y==k).sum())}" for k in range(3)))

    # ── 2. Split GROUPE : test 15% des groupes, puis val 15% ──────────────
    print("\n[2/4] Decoupage groupe par frame (aucune frame a cheval)...")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, te_idx = next(gss1.split(X, y, groups=g))
    Xtv, ytv, gtv = X[tv_idx], y[tv_idx], g[tv_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    tr_idx, va_idx = next(gss2.split(Xtv, ytv, groups=gtv))

    Xtr, ytr = Xtv[tr_idx], ytv[tr_idx]
    Xva, yva = Xtv[va_idx], ytv[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # verif anti-fuite
    s_tr = set(gtv[tr_idx]); s_va = set(gtv[va_idx]); s_te = set(g[te_idx])
    print(f"  train: {len(Xtr)} ROI / {len(s_tr)} frames")
    print(f"  val  : {len(Xva)} ROI / {len(s_va)} frames")
    print(f"  test : {len(Xte)} ROI / {len(s_te)} frames")
    print(f"  >> chevauchement train/test (doit etre 0) : {len(s_tr & s_te)}")
    print(f"  >> chevauchement val/test   (doit etre 0) : {len(s_va & s_te)}")
    print(f"  test par classe : " +
          ", ".join(f"{QUALITY[k]}={int((yte==k).sum())}" for k in range(3)))

    # ── 3. GridSearchCV GROUPE + stratifie (config identique a l'original) ─
    print("\n[3/4] GridSearchCV (StratifiedGroupKFold 5) sur train+val...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.001],
    }
    Xg = np.vstack([Xtr, Xva]); yg = np.concatenate([ytr, yva])
    gg = np.concatenate([gtv[tr_idx], gtv[va_idx]])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro",
                        n_jobs=-1, verbose=1, refit=True)
    grid.fit(Xg, yg, groups=gg)
    print(f"\n  Meilleurs hyperparametres : {grid.best_params_}")
    print(f"  F1-macro (CV groupe)      : {grid.best_score_:.4f}")

    # ── 4. Evaluation sur le test groupe (sans fuite) ─────────────────────
    print("\n[4/4] Evaluation sur le test GROUPE (jamais vu)...\n")
    yhat = grid.best_estimator_.predict(Xte)
    report = classification_report(yte, yhat, target_names=QUALITY, digits=4)
    acc = accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")
    print(report)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1m:.4f}")

    print("\n" + "=" * 64)
    print("  COMPARAISON")
    print("=" * 64)
    print(f"  Split d'origine (AVEC fuite)  : F1-macro = 0.9591  | acc = 0.9778")
    print(f"  Split groupe   (SANS fuite)   : F1-macro = {f1m:.4f}  | acc = {acc:.4f}")
    print(f"  Ecart F1-macro                : {(f1m-0.9591)*100:+.2f} points")

    (OUTPUT / "report_grouped.txt").write_text(
        f"SVM re-split GROUPE PAR FRAME (sans fuite)\n"
        f"Best params : {grid.best_params_}\n"
        f"F1-macro CV (groupe) : {grid.best_score_:.4f}\n\n"
        f"--- Test groupe ---\n{report}\n"
        f"Accuracy : {acc:.4f}\nF1-macro : {f1m:.4f}\n\n"
        f"Reference split d'origine (avec fuite) : F1-macro=0.9591 acc=0.9778\n"
        f"Ecart F1-macro : {(f1m-0.9591)*100:+.2f} points\n",
        encoding="utf-8")
    print(f"\n  Rapport sauvegarde : {OUTPUT / 'report_grouped.txt'}")


if __name__ == "__main__":
    main()
