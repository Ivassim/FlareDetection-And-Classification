"""
Évaluation externe — test de généralisation sur le dataset v1i.

Stratégie :
  Le modèle prédit 6 classes (Dark-Flare, Dark-Smoke, …).
  Le dataset v1i n'a que 2 classes : fire(0), smoke(1).

  Remapping des prédictions :
    Dark-Flare(0), Light-Flare(2), Medium-Flare(4)  → fire(0)
    Dark-Smoke(1), Light-Smoke(3), Medium-Smoke(5)  → smoke(1)

  On calcule ensuite le mAP à 2 classes contre les ground-truths v1i.

Usage :
    python -m src.evaluation.eval_external [--split test] [--conf 0.25]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from src.models.yolo_model import GasFlareDector
from src.utils.config import BEST_WEIGHTS, CONF_THRES, IOU_THRES, LOGS_DIR
from src.utils.helpers import ensure_dir, save_json, save_csv, timestamp

ROOT          = Path(__file__).resolve().parents[2]
EXTERNAL_DATA = ROOT / "data" / "external_test"

# Remapping 6 classes → 2 super-classes
# Dark-Flare=0, Dark-Smoke=1, Light-Flare=2, Light-Smoke=3, Medium-Flare=4, Medium-Smoke=5
REMAP = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}   # flare→0, smoke→1
SUPER_CLASSES = ["fire", "smoke"]


# ── Metric helpers ────────────────────────────────────────────────────────────

def box_iou_single(b1, b2) -> float:
    """IoU entre deux boîtes [x1 y1 x2 y2]."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(cx, cy, w, h, img_w=1.0, img_h=1.0):
    """Convertit cx cy w h (normalisé) → x1 y1 x2 y2 (pixels ou normalisé)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def compute_ap(recalls, precisions) -> float:
    """Average Precision via interpolation 101 points."""
    ap = 0.0
    for t in [i / 100 for i in range(101)]:
        p = max((precisions[i] for i, r in enumerate(recalls) if r >= t), default=0.0)
        ap += p / 101
    return ap


def compute_map(
    all_preds: dict[int, list],   # class_id → [(conf, tp), ...]
    n_gt: dict[int, int],         # class_id → nb GT
    iou_thresholds=None,
) -> dict:
    """
    Calcule mAP50 et mAP50-95 à partir des prédictions collectées.

    all_preds : pour chaque classe, liste de (confidence, is_tp)
                triée par confiance décroissante
    n_gt      : nombre total de GT par classe
    """
    if iou_thresholds is None:
        iou_thresholds = [0.50 + 0.05 * i for i in range(10)]

    # On stocke AP par classe par seuil IoU
    aps = defaultdict(list)

    for cls in sorted(set(all_preds) | set(n_gt)):
        preds = sorted(all_preds.get(cls, []), key=lambda x: -x[0])
        n     = n_gt.get(cls, 0)
        if n == 0:
            continue

        for iou_thr in iou_thresholds:
            tp_cumul = fp_cumul = 0
            recalls_pts = []
            precisions_pts = []
            for conf, tp_flag in preds:
                if tp_flag >= iou_thr:
                    tp_cumul += 1
                else:
                    fp_cumul += 1
                rec  = tp_cumul / n if n > 0 else 0
                prec = tp_cumul / (tp_cumul + fp_cumul)
                recalls_pts.append(rec)
                precisions_pts.append(prec)

            ap = compute_ap(recalls_pts, precisions_pts)
            aps[cls].append(ap)

    results = {}
    all_ap50    = []
    all_ap5095  = []
    for cls, ap_list in aps.items():
        ap50   = ap_list[0]   if ap_list else 0.0
        ap5095 = sum(ap_list) / len(ap_list) if ap_list else 0.0
        results[cls] = {"AP50": round(ap50, 4), "AP50_95": round(ap5095, 4)}
        all_ap50.append(ap50)
        all_ap5095.append(ap5095)

    results["ALL"] = {
        "mAP50"    : round(sum(all_ap50)   / len(all_ap50)   if all_ap50  else 0, 4),
        "mAP50_95" : round(sum(all_ap5095) / len(all_ap5095) if all_ap5095 else 0, 4),
    }
    return results


# ── Main evaluation ───────────────────────────────────────────────────────────

def load_gt(label_path: Path) -> list[dict]:
    """Charge les annotations GT d'un fichier label v1i (bbox)."""
    gt = []
    text = label_path.read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1, y1, x2, y2 = xywh_to_xyxy(cx, cy, w, h)
        gt.append({"cls": cls, "box": [x1, y1, x2, y2]})
    return gt


def evaluate_external(
    split: str = "test",
    weights: Path = BEST_WEIGHTS,
    conf: float = CONF_THRES,
    iou_match: float = 0.50,
    save_results: bool = True,
) -> dict:
    img_dir = EXTERNAL_DATA / split / "images"
    lbl_dir = EXTERNAL_DATA / split / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {img_dir}\n"
            "Lancez d'abord : python -m src.dataset.convert_v1i"
        )

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not images:
        raise ValueError(f"Aucune image trouvée dans {img_dir}")

    print(f"[INFO] Évaluation externe — split '{split}'")
    print(f"[INFO] {len(images)} images  |  Conf={conf}  IoU_match={iou_match}")
    print(f"[INFO] Modèle : {weights}")

    detector = GasFlareDector(weights=weights, conf=conf, iou=IOU_THRES)

    # Structures pour le calcul de mAP
    # Pour chaque super-classe, pour chaque IoU-seuil on collecte (conf, best_iou)
    all_preds : dict[int, list] = defaultdict(list)  # cls → [(conf, best_iou)]
    n_gt      : dict[int, int]  = defaultdict(int)   # cls → nb GT total

    per_image_rows = []

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt(lbl_path) if lbl_path.exists() else []

        # Inférence
        result = detector.predict(source=str(img_path), save=False)[0]

        # Prédictions remappées en 2 classes
        # xyxyn = coordonnées normalisées (0-1), même espace que les GT
        preds = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                orig_cls = int(box.cls.item())
                mapped   = REMAP.get(orig_cls, -1)
                if mapped < 0:
                    continue
                conf_val = float(box.conf.item())
                xyxy     = box.xyxyn[0].tolist()   # normalisé [0-1]
                preds.append({"cls": mapped, "conf": conf_val, "box": xyxy})

        # Compter GT par classe
        for g in gt_boxes:
            n_gt[g["cls"]] += 1

        # Associer prédictions aux GT (greedy, par confiance décroissante)
        gt_matched = [False] * len(gt_boxes)
        for pred in sorted(preds, key=lambda x: -x["conf"]):
            best_iou  = 0.0
            best_idx  = -1
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                if gt["cls"] != pred["cls"]:
                    continue
                iou = box_iou_single(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_idx >= 0 and best_iou >= iou_match:
                gt_matched[best_idx] = True

            all_preds[pred["cls"]].append((pred["conf"], best_iou))

        per_image_rows.append({
            "image"   : img_path.name,
            "n_gt"    : len(gt_boxes),
            "n_pred"  : len(preds),
            "n_matched": sum(gt_matched),
        })

    # ── Calcul mAP ────────────────────────────────────────────────────────────
    metrics = compute_map(all_preds, n_gt)

    # ── Affichage ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Généralisation externe (v1i) — résultats")
    print("=" * 55)
    print(f"  {'mAP@0.50':<28}  {metrics['ALL']['mAP50']:.4f}")
    print(f"  {'mAP@0.50:0.95':<28}  {metrics['ALL']['mAP50_95']:.4f}")
    print("-" * 55)
    print(f"  {'Super-classe':<20}  {'AP50':>8}  {'AP50-95':>8}")
    print("-" * 55)
    for i, name in enumerate(SUPER_CLASSES):
        m = metrics.get(i, {"AP50": 0, "AP50_95": 0})
        print(f"  {name:<20}  {m['AP50']:>8.4f}  {m['AP50_95']:>8.4f}")
    print("=" * 55)

    result = {
        "split"      : split,
        "weights"    : str(weights),
        "conf_thres" : conf,
        "iou_match"  : iou_match,
        "n_images"   : len(images),
        "n_gt_fire"  : n_gt.get(0, 0),
        "n_gt_smoke" : n_gt.get(1, 0),
        "mAP50"      : metrics["ALL"]["mAP50"],
        "mAP50_95"   : metrics["ALL"]["mAP50_95"],
        "AP_fire"    : metrics.get(0, {"AP50": 0, "AP50_95": 0}),
        "AP_smoke"   : metrics.get(1, {"AP50": 0, "AP50_95": 0}),
        "per_image"  : per_image_rows,
    }

    if save_results:
        ts       = timestamp()
        log_dir  = ensure_dir(LOGS_DIR)
        json_path = log_dir / f"eval_external_{split}_{ts}.json"
        csv_path  = log_dir / f"eval_external_{split}_{ts}.csv"
        save_json({k: v for k, v in result.items() if k != "per_image"}, json_path)
        save_csv(per_image_rows, csv_path)
        print(f"\n[INFO] JSON : {json_path}")
        print(f"[INFO] CSV  : {csv_path}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation externe sur v1i")
    parser.add_argument("--split",  default="test", choices=["test", "valid", "train"])
    parser.add_argument("--weights", default=str(BEST_WEIGHTS))
    parser.add_argument("--conf",    type=float, default=CONF_THRES)
    parser.add_argument("--iou",     type=float, default=0.50, help="Seuil IoU pour TP/FP")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    evaluate_external(
        split=args.split,
        weights=Path(args.weights),
        conf=args.conf,
        iou_match=args.iou,
        save_results=not args.no_save,
    )
