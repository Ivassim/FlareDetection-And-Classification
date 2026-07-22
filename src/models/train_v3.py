"""
Training script v3 – Gas Flare Detection (split 3-way rigoureux).

Différence clé par rapport à v2 :
  - Utilise data3.yaml : train3(70%) / val3(15%) / test3(15%)
  - Le test3 n'est JAMAIS vu pendant l'entraînement ni le model selection
  - Les hyperparamètres sont identiques à v2 (déjà optimisés)

Prérequis :
    python -m src.dataset.resplit3way   # génère train3/ val3/ test3/ data3.yaml

Usage :
    python src/models/train_v3.py
    python src/models/train_v3.py --model yolo11m.pt --run-name gas_flare_yolo11m_v1
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DATA_YAML  = ROOT / "Gas Flaring Detection.v15i.yolov8" / "data3.yaml"
OUTPUT_DIR = ROOT / "outputs" / "models"

# ── Config (défauts migration YOLO11m) ───────────────────────────────────────
MODEL_VARIANT = "yolo11m.pt"
RUN_NAME      = "gas_flare_yolo11m_v1"
EPOCHS        = 170
IMG_SIZE      = 640
BATCH_SIZE    = 16
PATIENCE      = 40
WORKERS       = 4
DEVICE        = 0
FREEZE_LAYERS = 10


def train(
    model_variant: str = MODEL_VARIANT,
    run_name: str = RUN_NAME,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: int | str = DEVICE,
):
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"YAML introuvable : {DATA_YAML}\n"
            "Lancez d'abord : python -m src.dataset.resplit3way"
        )

    print(f"[INFO] Modèle      : {model_variant}")
    print(f"[INFO] Dataset     : {DATA_YAML}")
    print(f"[INFO] Output dir  : {OUTPUT_DIR}")
    print(f"[INFO] Run name    : {run_name}")
    print(f"[INFO] Split       : train3(70%) / val3(15%) / test3(15%)")
    print(f"[INFO] Epochs      : {epochs}  |  Patience : {PATIENCE}")

    model = YOLO(model_variant)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch_size,
        patience=PATIENCE,
        workers=WORKERS,
        device=device,

        # ── Répertoire de sortie ───────────────────────────────────────────
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True,

        # ── Scheduler ─────────────────────────────────────────────────────
        cos_lr=True,
        lr0=0.01,
        lrf=0.005,
        warmup_epochs=5.0,

        # ── Régularisation ────────────────────────────────────────────────
        weight_decay=0.0005,
        dropout=0.0,
        label_smoothing=0.05,

        # ── Freeze backbone ───────────────────────────────────────────────
        freeze=FREEZE_LAYERS,

        # ── Augmentations couleur ─────────────────────────────────────────
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,

        # ── Augmentations géométriques ────────────────────────────────────
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,

        # ── Augmentations avancées ────────────────────────────────────────
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.4,
        close_mosaic=15,

        # ── Logs / sauvegardes ────────────────────────────────────────────
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best = OUTPUT_DIR / run_name / "weights" / "best.pt"
    print("\n[INFO] Entraînement terminé.")
    print(f"[INFO] Meilleurs poids : {best}")
    print(f"\n[INFO] Prochaine étape — évaluation sur le test set (jamais vu) :")
    print(f"       python main.py evaluate --weights {best} --split test")
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Training YOLO11m (split 3-way)")
    parser.add_argument("--model", default=MODEL_VARIANT, help="Modèle de base Ultralytics (.pt)")
    parser.add_argument("--run-name", default=RUN_NAME, help="Nom du dossier de sortie dans outputs/models")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", default=DEVICE, help="Device (ex: 0, 1, cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        model_variant=args.model,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
    )
