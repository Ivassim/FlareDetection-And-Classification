"""
Script de re-split stratifié du dataset.

Problème identifié :
  - Train : Dark-Smoke 39%, Dark-Flare 29% (dominants)
  - Valid  : Light-Flare 35%, Medium-Smoke 23% (distribution très différente)

Ce script fusionne train+valid et re-divise en 80/20 stratifié par classe,
garantissant que chaque split reflète la même distribution de classes.

Usage :
    python -m src.dataset.resplit [--ratio 0.8] [--seed 42]

Sortie :
    Gas Flaring Detection.v15i.yolov8/
        train_balanced/  images/ + labels/
        valid_balanced/  images/ + labels/
        data_balanced.yaml
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DATASET   = ROOT / "Gas Flaring Detection.v15i.yolov8"
CLASS_NAMES = [
    "Dark-Flare", "Dark-Smoke", "Light-Flare",
    "Light-Smoke", "Medium-Flare", "Medium-Smoke",
]


def collect_samples(splits=("train", "valid")) -> list[dict]:
    """
    Rassemble tous les fichiers image+label de plusieurs splits.
    Retourne une liste de dicts {image, label, primary_class}.
    """
    samples = []
    for split in splits:
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        if not img_dir.exists():
            print(f"[WARN] Dossier introuvable : {img_dir}")
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            # Classe primaire = classe la plus fréquente dans l'image
            cls_counts = defaultdict(int)
            for line in lbl.read_text().strip().splitlines():
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1
            if not cls_counts:
                continue
            primary = max(cls_counts, key=cls_counts.get)
            samples.append({"image": img, "label": lbl, "primary_class": primary})
    return samples


def stratified_split(samples: list[dict], train_ratio: float, seed: int):
    """Divise en train/valid en preservant la distribution par classe."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["primary_class"]].append(s)

    train_samples, valid_samples = [], []
    for cls, items in sorted(by_class.items()):
        random.seed(seed + cls)
        random.shuffle(items)
        n_train = max(1, int(len(items) * train_ratio))
        train_samples.extend(items[:n_train])
        valid_samples.extend(items[n_train:])

    return train_samples, valid_samples


def copy_samples(samples: list[dict], dest_img: Path, dest_lbl: Path):
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)
    for s in samples:
        shutil.copy2(s["image"], dest_img / s["image"].name)
        shutil.copy2(s["label"], dest_lbl / s["label"].name)


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


def resplit(train_ratio: float = 0.80, seed: int = 42):
    print("[INFO] Collecte des images train+valid...")
    samples = collect_samples(splits=("train", "valid"))
    print(f"[INFO] Total : {len(samples)} images")

    train_s, valid_s = stratified_split(samples, train_ratio, seed)

    print_distribution("Train (avant)", [s for s in samples if (DATASET/"train"/"images"/s["image"].name).exists()])
    print_distribution("Valid (avant)", [s for s in samples if (DATASET/"valid"/"images"/s["image"].name).exists()])
    print_distribution("Train (après re-split)", train_s)
    print_distribution("Valid (après re-split)", valid_s)

    # Copie vers train_balanced / valid_balanced
    for split, slist in [("train_balanced", train_s), ("valid_balanced", valid_s)]:
        dst_img = DATASET / split / "images"
        dst_lbl = DATASET / split / "labels"
        if dst_img.exists():
            shutil.rmtree(dst_img.parent)
        copy_samples(slist, dst_img, dst_lbl)
        print(f"\n[INFO] {split} → {len(slist)} images copiées dans {DATASET/split}")

    # Génère data_balanced.yaml
    orig_yaml = (DATASET / "data.yaml").read_text()
    balanced_yaml = orig_yaml
    balanced_yaml = balanced_yaml.replace("train: ../train/images",   "train: ../train_balanced/images")
    balanced_yaml = balanced_yaml.replace("val: ../valid/images",     "val: ../valid_balanced/images")
    out_yaml = DATASET / "data_balanced.yaml"
    out_yaml.write_text(balanced_yaml)
    print(f"\n[INFO] YAML généré : {out_yaml}")
    print("\n[INFO] Re-split terminé. Pour utiliser le nouveau split :")
    print(f"       Remplacer DATA_YAML par : {out_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-split stratifié du dataset")
    parser.add_argument("--ratio", type=float, default=0.80, help="Part du train (défaut 0.80)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Graine aléatoire")
    args = parser.parse_args()
    resplit(train_ratio=args.ratio, seed=args.seed)
