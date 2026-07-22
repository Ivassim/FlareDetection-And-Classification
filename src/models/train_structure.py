"""
Training script STRUCTURE – détection de l'installation de torchère (flare stack).

Objectif : entraîner un YOLO11-n auxiliaire qui détecte la STRUCTURE physique
(mât de torchère, cheminée) et non la flamme. Utilisé par le pipeline pour
valider qu'une flamme détectée surmonte bien une installation industrielle
(rejet des fausses torchères type briquet/bougie en webcam).

Dataset  : u2_eo_train v12 (FireDetection, Ulsan — CC BY 4.0)
           5 classes : chimney(0), fire smoke(1), flame(2),
                       flare stack(3), normal smoke(4)
           12 383 train / 5 026 valid / 241 test
Le modèle principal YOLO11-m (gas_flare_yolo11m_v1) n'est PAS modifié.

Hyperparamètres : identiques à train_v3.py (recette validée du mémoire),
durée estimée sur RTX 3060 Laptop : ~4-6 h pour 80 epochs.

Usage :
    python src/models/train_structure.py
    python src/models/train_structure.py --epochs 100 --batch 32
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DATA_YAML  = (ROOT.parent / "u2_eo_train.v12-u2_eo_v8_debug_2.yolov8"
              / "data_structure.yaml")
OUTPUT_DIR = ROOT / "outputs" / "models"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_VARIANT = str(ROOT / "yolo11n.pt")   # poids COCO déjà téléchargés (offline OK)
RUN_NAME      = "flare_structure_yolo11n_v1"
EPOCHS        = 80
IMG_SIZE      = 640
BATCH_SIZE    = 16
PATIENCE      = 20
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
            "Vérifiez que le dataset u2_eo_train.v12 est bien dans PFE - Copie."
        )
    if not Path(model_variant).exists():
        raise FileNotFoundError(
            f"Poids de base introuvables : {model_variant}\n"
            "yolo11n.pt doit être à la racine de PFE-Object-Detection."
        )

    print(f"[INFO] Modèle      : {model_variant}")
    print(f"[INFO] Dataset     : {DATA_YAML}")
    print(f"[INFO] Output dir  : {OUTPUT_DIR}")
    print(f"[INFO] Run name    : {run_name}")
    print(f"[INFO] Classes     : chimney / fire smoke / flame / flare stack / normal smoke")
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

        # ── Scheduler (recette train_v3) ──────────────────────────────────
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
        # (degrees réduit vs train_v3 : les mâts sont des objets fins
        #  verticaux, une rotation forte dégrade les boîtes)
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,

        # ── Augmentations avancées ────────────────────────────────────────
        # (mixup/copy_paste coupés : le dataset Roboflow contient déjà des
        #  copies augmentées avec bruit incrusté, on évite d'empiler)
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        close_mosaic=10,

        # ── Logs / sauvegardes ────────────────────────────────────────────
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best = OUTPUT_DIR / run_name / "weights" / "best.pt"
    print("\n[INFO] Entraînement terminé.")
    print(f"[INFO] Meilleurs poids : {best}")
    print("\n[INFO] Prochaine étape — test d'écart de domaine sur vos vidéos :")
    print("       python eval_structure_domaine.py")
    print("\n[INFO] Le pipeline (FlareProcessor / app_gui) détectera ces poids")
    print("       automatiquement au prochain lancement de l'interface.")
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Training YOLO11-n structure (flare stack / chimney)")
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
