"""
Re-split stratifié en 3 parties : train / val / test.

Problème avec un split 2-way (train/val) pour le PFE :
  - Le valid_balanced a influencé le choix de best.pt via early stopping.
  - Il ne peut donc pas servir de test set indépendant.

Ce script fusionne train_balanced + valid_balanced et re-divise en 3 :
  - train (70%) → mise à jour des poids
  - val   (15%) → early stopping / sélection du meilleur modèle
  - test  (15%) → évaluation finale, jamais vu pendant l'entraînement

Sortie :
  Gas Flaring Detection.v15i.yolov8/
      train3/   images/ + labels/
      val3/     images/ + labels/
      test3/    images/ + labels/
      data3.yaml

Usage :
    python -m src.dataset.resplit3way [--train 0.70] [--val 0.15] [--seed 42]
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Gas Flaring Detection.v15i.yolov8"
CLASS_NAMES = [
    "Dark-Flare", "Dark-Smoke", "Light-Flare",
    "Light-Smoke", "Medium-Flare", "Medium-Smoke",
]


def collect_samples(splits=("train_balanced", "valid_balanced")) -> list[dict]:
    """Rassemble tous les fichiers image+label des splits indiqués."""
    samples = []
    for split in splits:
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        if not img_dir.exists():
            print(f"  [WARN] Introuvable : {img_dir}")
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            cls_counts = defaultdict(int)
            for line in lbl.read_text().strip().splitlines():
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1
            if not cls_counts:
                continue
            primary = max(cls_counts, key=cls_counts.get)
            samples.append({"image": img, "label": lbl, "primary_class": primary})
    return samples


def stratified_3way_split(
    samples: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Divise en train/val/test en préservant la distribution par classe."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["primary_class"]].append(s)

    train_s, val_s, test_s = [], [], []
    for cls, items in sorted(by_class.items()):
        random.seed(seed + cls)
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # S'assurer qu'il reste au moins 1 sample pour le test
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train_s.extend(items[:n_train])
        val_s.extend(items[n_train:n_train + n_val])
        test_s.extend(items[n_train + n_val:])

    return train_s, val_s, test_s


def copy_samples(samples: list[dict], dst: Path):
    img_dir = dst / "images"
    lbl_dir = dst / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        shutil.copy2(s["image"], img_dir / s["image"].name)
        shutil.copy2(s["label"], lbl_dir / s["label"].name)


def print_distribution(label: str, samples: list[dict]):
    counts = defaultdict(int)
    for s in samples:
        counts[s["primary_class"]] += 1
    total = len(samples)
    print(f"\n  {label} ({total} images) :")
    for i, name in enumerate(CLASS_NAMES):
        n = counts.get(i, 0)
        bar = "█" * int(n / total * 30) if total else ""
        print(f"    {name:<18}: {n:>4}  ({n/total*100:5.1f}%)  {bar}")


def generate_yaml():
    content = """train: ../train3/images
val:   ../val3/images
test:  ../test3/images

nc: 6
names: ['Dark-Flare', 'Dark-Smoke', 'Light-Flare', 'Light-Smoke', 'Medium-Flare', 'Medium-Smoke']

roboflow:
  workspace: esraa-ramadan
  project: gas-flaring-detection
  version: 15
  license: CC BY 4.0
  url: https://universe.roboflow.com/esraa-ramadan/gas-flaring-detection/dataset/15
"""
    path = DATASET / "data3.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def resplit3way(train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"[INFO] Split : train={train_ratio:.0%}  val={val_ratio:.0%}  test={test_ratio:.0%}")
    print("[INFO] Collecte des images depuis train_balanced + valid_balanced...")

    samples = collect_samples()
    print(f"[INFO] Total : {len(samples)} images")

    train_s, val_s, test_s = stratified_3way_split(samples, train_ratio, val_ratio, seed)

    print_distribution("train3", train_s)
    print_distribution("val3",   val_s)
    print_distribution("test3",  test_s)

    # Nettoyage des anciens dossiers si existants
    for split in ("train3", "val3", "test3"):
        dst = DATASET / split
        if dst.exists():
            shutil.rmtree(dst)

    copy_samples(train_s, DATASET / "train3")
    copy_samples(val_s,   DATASET / "val3")
    copy_samples(test_s,  DATASET / "test3")

    yaml_path = generate_yaml()

    print(f"\n[INFO] Résumé :")
    print(f"  train3 : {len(train_s)} images")
    print(f"  val3   : {len(val_s)} images")
    print(f"  test3  : {len(test_s)} images")
    print(f"\n[INFO] YAML généré : {yaml_path}")
    print("\n[INFO] Re-split 3-way terminé.")
    print("       Prochaine étape : lancer src/models/train_v3.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-split stratifié 3-way")
    parser.add_argument("--train", type=float, default=0.70, help="Part train (défaut 0.70)")
    parser.add_argument("--val",   type=float, default=0.15, help="Part val   (défaut 0.15)")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()
    resplit3way(train_ratio=args.train, val_ratio=args.val, seed=args.seed)
