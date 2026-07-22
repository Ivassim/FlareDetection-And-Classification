"""
Conversion du dataset v1i (segmentation polygone, 2 classes) vers
le format YOLO bbox (5 valeurs) compatible avec une évaluation externe.

Source  : flaring-gas-detection.v1i.yolov8/  (fire=0, smoke=1)
Sortie  : data/external_test/                 (même mapping fire=0, smoke=1)
YAML    : data/external_test/data_external.yaml

Format polygone source :
    class x1 y1 x2 y2 ... xN yN   (coordonnées normalisées)

Conversion → bbox :
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w  = xmax - xmin
    h  = ymax - ymin

Usage :
    python -m src.dataset.convert_v1i [--splits test valid train]
"""

import argparse
import shutil
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
SRC      = ROOT / "flaring-gas-detection.v1i.yolov8"
DST      = ROOT / "data" / "external_test"

CLASS_NAMES = ["fire", "smoke"]   # classes v1i originales


def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Convertit une liste de points polygone en cx cy w h (normalisés)."""
    xs = values[0::2]
    ys = values[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w  = xmax - xmin
    h  = ymax - ymin
    # Clip pour rester dans [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def convert_label_file(src_lbl: Path, dst_lbl: Path) -> int:
    """
    Convertit un fichier label polygone → bbox.
    Retourne le nombre d'annotations converties.
    """
    lines = src_lbl.read_text(encoding="utf-8").strip().splitlines()
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) < 4:
            # Ligne invalide ou déjà en format bbox (4 valeurs)
            continue

        if len(coords) == 4:
            # Déjà au format bbox cx cy w h
            cx, cy, w, h = coords
        else:
            # Format polygone → convertir
            if len(coords) % 2 != 0:
                coords = coords[:-1]   # enlever valeur orpheline si nécessaire
            cx, cy, w, h = polygon_to_bbox(coords)

        out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines)


def convert_split(split: str) -> dict:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"
    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

    if not src_img.exists():
        print(f"  [WARN] Split '{split}' introuvable dans {SRC}, ignoré.")
        return {}

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "annotations": 0, "fire": 0, "smoke": 0}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img in src_img.iterdir():
        if img.suffix.lower() not in exts:
            continue
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            continue

        # Copier l'image
        shutil.copy2(img, dst_img / img.name)

        # Convertir le label
        n = convert_label_file(lbl, dst_lbl / lbl.name)
        stats["images"] += 1
        stats["annotations"] += n

        # Compter par classe
        for line in (dst_lbl / lbl.name).read_text().strip().splitlines():
            if line.strip():
                c = int(line.split()[0])
                if c == 0: stats["fire"] += 1
                elif c == 1: stats["smoke"] += 1

    return stats


def generate_yaml():
    yaml_content = f"""train: ../data/external_test/train/images
val:   ../data/external_test/valid/images
test:  ../data/external_test/test/images

nc: 2
names: ['fire', 'smoke']

# Généré automatiquement par src/dataset/convert_v1i.py
# Source : flaring-gas-detection.v1i.yolov8 (polygone → bbox)
"""
    yaml_path = DST / "data_external.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def convert(splits=("test", "valid", "train")):
    print(f"[INFO] Source : {SRC}")
    print(f"[INFO] Dest   : {DST}")
    print()

    all_stats = {}
    for split in splits:
        print(f"[INFO] Conversion split '{split}'...")
        stats = convert_split(split)
        if stats:
            all_stats[split] = stats
            print(f"  {stats['images']} images  |  {stats['annotations']} annotations "
                  f"  (fire={stats['fire']}, smoke={stats['smoke']})")

    yaml_path = generate_yaml()
    print(f"\n[INFO] YAML généré : {yaml_path}")
    print("[INFO] Conversion terminée.")
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversion v1i polygone → bbox")
    parser.add_argument(
        "--splits", nargs="+",
        default=["test", "valid", "train"],
        help="Splits à convertir (défaut: test valid train)"
    )
    args = parser.parse_args()
    convert(splits=args.splits)
