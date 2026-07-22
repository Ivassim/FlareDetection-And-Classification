"""
Extraction automatique des ROI depuis le dataset v15i.

Stratégie de labellisation (sans expert) :
  Les 6 classes YOLO contiennent déjà l'information de qualité de combustion.
  
  Mapping :
    Light-Flare (2), Light-Smoke (3)     →  "bonne"    (combustion complète)
    Medium-Flare (4), Medium-Smoke (5)   →  "moyenne"  (combustion partielle)
    Dark-Flare (0), Dark-Smoke (1)       →  "mauvaise" (combustion incomplète)

Sortie :
    data/rois/{split}/{quality}/  ← images ROI cropées
    data/rois/manifest.csv        ← inventaire complet

Usage :
    python -m src.classification.extract_rois
"""

from pathlib import Path
from collections import defaultdict
import csv
import cv2

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Gas Flaring Detection.v15i.yolov8"

# ── Mapping classes → qualité ────────────────────────────────────────────

CLASS_NAMES = [
    "Dark-Flare", "Dark-Smoke", "Light-Flare",
    "Light-Smoke", "Medium-Flare", "Medium-Smoke",
]

QUALITY_MAP = {
    0: "mauvaise",   # Dark-Flare
    1: "mauvaise",   # Dark-Smoke
    2: "bonne",      # Light-Flare
    3: "bonne",      # Light-Smoke
    4: "moyenne",    # Medium-Flare
    5: "moyenne",    # Medium-Smoke
}

SPLITS = ["train3", "val3", "test3"]


def xywh_to_xyxy_pixels(cx, cy, w, h, img_w, img_h):
    """Convertit YOLO normalisé (cx, cy, w, h) → (x1, y1, x2, y2) pixels."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # Clamp aux bords de l'image
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


def extract_rois_for_split(split: str, output_dir: Path, min_size: int = 32):
    """
    Pour chaque image d'un split :
      1. Lire les annotations GT (bbox)
      2. Cropper chaque bbox → ROI
      3. Sauvegarder dans output_dir/{split}/{quality}/

    Parameters
    ----------
    split : str
        Nom du split (train3, val3, test3).
    output_dir : Path
        Répertoire de sortie racine.
    min_size : int
        Taille minimale (px) d'un côté de la ROI pour l'accepter.

    Returns
    -------
    stats : dict
        {quality_name: count}
    manifest : list[dict]
        Métadonnées de chaque ROI extraite.
    """
    img_dir = DATASET / split / "images"
    lbl_dir = DATASET / split / "labels"

    if not img_dir.exists():
        print(f"  [WARN] Dossier introuvable : {img_dir}")
        return {}, []

    stats = defaultdict(int)
    manifest = []
    skipped_small = 0
    skipped_read = 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in image_exts)

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped_read += 1
            continue
        h_img, w_img = img.shape[:2]

        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id not in QUALITY_MAP:
                continue

            cx, cy, w, h = map(float, parts[1:5])
            quality = QUALITY_MAP[cls_id]

            x1, y1, x2, y2 = xywh_to_xyxy_pixels(cx, cy, w, h, w_img, h_img)

            # Filtrer les ROI trop petites
            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w < min_size or roi_h < min_size:
                skipped_small += 1
                continue

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Sauvegarder la ROI
            out_dir = output_dir / split / quality
            out_dir.mkdir(parents=True, exist_ok=True)
            roi_name = f"{img_path.stem}_roi{i}.jpg"
            cv2.imwrite(str(out_dir / roi_name), roi)

            manifest.append({
                "filename": roi_name,
                "quality": quality,
                "original_class": cls_id,
                "original_class_name": CLASS_NAMES[cls_id],
                "split": split,
                "roi_width": roi_w,
                "roi_height": roi_h,
            })
            stats[quality] += 1

    if skipped_small:
        print(f"  [INFO] {skipped_small} ROI ignorées (trop petites < {min_size}px)")
    if skipped_read:
        print(f"  [WARN] {skipped_read} images non lisibles")

    return dict(stats), manifest


def main():
    output_dir = ROOT / "data" / "rois"

    print("=" * 60)
    print("  Extraction des ROI pour classification qualité")
    print("=" * 60)
    print(f"  Source  : {DATASET}")
    print(f"  Sortie  : {output_dir}")
    print(f"  Mapping : {QUALITY_MAP}")
    print()

    all_manifest = []
    global_stats = defaultdict(int)

    for split in SPLITS:
        print(f"[INFO] Traitement split '{split}'...")
        stats, manifest = extract_rois_for_split(split, output_dir)
        all_manifest.extend(manifest)
        for q, n in sorted(stats.items()):
            global_stats[q] += n
            print(f"  {q:<12}: {n:>5} ROI")

    # Sauvegarder le manifeste complet
    csv_path = output_dir / "manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "quality", "original_class", "original_class_name",
                      "split", "roi_width", "roi_height"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_manifest)

    # Résumé
    print("\n" + "=" * 60)
    print("  Résumé global")
    print("=" * 60)
    for q in ["bonne", "moyenne", "mauvaise"]:
        n = global_stats.get(q, 0)
        bar = "█" * int(n / max(global_stats.values(), default=1) * 30)
        print(f"  {q:<12}: {n:>5} ROI  {bar}")
    print(f"\n  Total       : {len(all_manifest):>5} ROI")
    print(f"  Manifeste   : {csv_path}")
    print("\n[DONE] Extraction terminée.")
    print("       Prochaine étape : python -m src.classification.train_svm")


if __name__ == "__main__":
    main()
