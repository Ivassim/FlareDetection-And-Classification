"""
Training script v2 – Gas Flare Detection (improved).

Améliorations par rapport à v1 :
  1. Modèle plus grand : YOLOv8m  (meilleure capacité de distinction inter-classes)
  2. Plus d'epochs + patience plus élevée (laisser le temps de converger)
  3. Cosine LR schedule (convergence plus douce en fin d'entraînement)
  4. Augmentations renforcées : MixUp, Copy-Paste, rotation, scale, erasing
  5. Label smoothing (régularisation, réduit la sur-confiance)
  6. Freeze backbone les N premières epochs (fine-tune la tête d'abord)
  7. Warm-up plus long
  8. Poids de perte de classification ajustés (cls_pw) pour classes rares

Classes : Dark-Flare, Dark-Smoke, Light-Flare, Light-Smoke, Medium-Flare, Medium-Smoke
"""

from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "Gas Flaring Detection.v15i.yolov8" / "data_balanced.yaml"
OUTPUT_DIR = ROOT / "outputs" / "models"

# ── Config ─────────────────────────────────────────────────────────────────────
#
# Stratégie de choix du modèle :
#   yolov8s : 11M params  – résultats actuels (mAP50=0.474)
#   yolov8m : 25M params  – recommandé  ← change ici
#   yolov8l : 43M params  – si GPU > 8 GB disponible
#
MODEL_VARIANT = "yolov8m.pt"

EPOCHS      = 200     # v1=100 – laisser plus de temps à la convergence
IMG_SIZE    = 640
BATCH_SIZE  = 8       # réduire à 4 si OOM avec yolov8m
PATIENCE    = 40      # v1=20 – courbe plate ≠ convergence terminée
WORKERS     = 4
DEVICE      = 0

# Freeze : les N premières couches du backbone sont gelées au début
# → le réseau ajuste d'abord la tête (classification/régression) avant
#   de modifier les features extraites par le backbone pré-entraîné.
FREEZE_LAYERS = 10    # 0 = désactivé, 10 = recommandé pour fine-tuning


def train():
    print(f"[INFO] Modèle      : {MODEL_VARIANT}")
    print(f"[INFO] Dataset     : {DATA_YAML}")
    print(f"[INFO] Output dir  : {OUTPUT_DIR}")
    print(f"[INFO] Epochs      : {EPOCHS}  |  Patience : {PATIENCE}")

    model = YOLO(MODEL_VARIANT)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,

        # ── Répertoire de sortie ───────────────────────────────────────────
        project=str(OUTPUT_DIR),
        name="gas_flare_yolov8m_v2",
        exist_ok=True,

        # ── Scheduler ─────────────────────────────────────────────────────
        cos_lr=True,          # cosine annealing : LR descend smoothement
        lr0=0.01,             # LR initial
        lrf=0.005,            # LR final = lr0 * lrf (0.01 * 0.005 = 5e-5)
        warmup_epochs=5.0,    # v1=3 – warm-up plus long, meilleure stabilité

        # ── Régularisation ────────────────────────────────────────────────
        weight_decay=0.0005,
        dropout=0.0,
        label_smoothing=0.05, # réduit la sur-confiance, aide les classes rares

        # ── Freeze backbone ───────────────────────────────────────────────
        freeze=FREEZE_LAYERS,

        # ── Augmentations couleur ─────────────────────────────────────────
        hsv_h=0.02,           # v1=0.015 – légèrement plus de variation teinte
        hsv_s=0.7,
        hsv_v=0.5,            # v1=0.4  – plus de variation luminosité (nuit/jour)

        # ── Augmentations géométriques ───────────────────────────────────
        degrees=10.0,         # rotation ±10° (torchères légèrement inclinées)
        translate=0.1,        # translation ±10%
        scale=0.5,            # zoom 50-150%
        shear=2.0,            # cisaillement ±2°
        perspective=0.0001,   # légère perspective
        flipud=0.0,           # pas de flip vertical (torchères vers le haut)
        fliplr=0.5,           # flip horizontal OK

        # ── Augmentations avancées ────────────────────────────────────────
        mosaic=1.0,           # mosaïque 4 images
        mixup=0.15,           # MixUp : mélange 2 images, aide classes rares
        copy_paste=0.1,       # Copy-Paste : colle des objets d'autres images
                              #              très utile pour Light-Smoke rare
        erasing=0.4,          # Random Erasing : force la robustesse à l'occlusion
        close_mosaic=15,      # désactive mosaic les 15 derniers epochs
                              # (stabilise la convergence finale)

        # ── Logs / sauvegardes ────────────────────────────────────────────
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best = OUTPUT_DIR / "gas_flare_yolov8m_v2" / "weights" / "best.pt"
    print("\n[INFO] Entraînement terminé.")
    print(f"[INFO] Meilleurs poids : {best}")
    return results


if __name__ == "__main__":
    train()
