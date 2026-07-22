"""
Training script for Gas Flare Detection using YOLOv8.
Dataset: Gas Flaring Detection v15 (6 classes)
Classes: Dark-Flare, Dark-Smoke, Light-Flare, Light-Smoke, Medium-Flare, Medium-Smoke
"""

from pathlib import Path
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # PFE-Object-Detection/
DATA_YAML = ROOT / "Gas Flaring Detection.v15i.yolov8" / "data.yaml"
OUTPUT_DIR = ROOT / "outputs" / "models"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_VARIANT = "yolov8s.pt"   # s=small : bon compromis précision/vitesse pour PFE
                                # options : yolov8n / yolov8s / yolov8m / yolov8l
EPOCHS        = 100
IMG_SIZE      = 640
BATCH_SIZE    = 8              # réduire à 8 si OOM (mémoire GPU insuffisante)
PATIENCE      = 20              # early stopping si pas d'amélioration pendant 20 epochs
WORKERS       = 4
DEVICE        = 0               # 0 = GPU (CUDA), 'cpu' = CPU

# ── Training ─────────────────────────────────────────────────────────────────
def train():
    print(f"[INFO] Loading model : {MODEL_VARIANT}")
    print(f"[INFO] Dataset       : {DATA_YAML}")
    print(f"[INFO] Output dir    : {OUTPUT_DIR}")

    model = YOLO(MODEL_VARIANT)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,

        # Répertoire de sortie
        project=str(OUTPUT_DIR),
        name="gas_flare_yolov8s",
        exist_ok=True,

        # Augmentations (utiles pour robustesse)
        hsv_h=0.015,     # variation teinte  (flammes changent de couleur)
        hsv_s=0.7,       # variation saturation
        hsv_v=0.4,       # variation luminosité (utile nuit/jour)
        flipud=0.0,      # pas de flip vertical (torchères sont toujours vers le haut)
        fliplr=0.5,      # flip horizontal OK
        mosaic=1.0,      # mosaïque 4 images (améliore détection multi-échelle)

        # Logs
        save=True,
        save_period=10,       # sauvegarde checkpoint toutes les 10 epochs
        plots=True,           # génère les courbes P/R/mAP
        verbose=True,
    )

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best weights saved at : {OUTPUT_DIR / 'gas_flare_yolov8s' / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    train()
