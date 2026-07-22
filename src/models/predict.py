"""
Inference script – run predictions on images, a folder, or a video.

Usage (standalone):
    python -m src.models.predict --source <path> [--conf 0.25] [--save]
"""

import argparse
from pathlib import Path

from src.models.yolo_model import GasFlareDector
from src.utils.config import BEST_WEIGHTS, CONF_THRES, IOU_THRES, PREDICTIONS_DIR
from src.utils.helpers import ensure_dir, save_json, timestamp


def predict(
    source: str | Path,
    weights: str | Path = BEST_WEIGHTS,
    conf: float = CONF_THRES,
    iou: float = IOU_THRES,
    save_images: bool = True,
    save_json_results: bool = True,
) -> list:
    """
    Run inference and optionally save annotated images + JSON summary.

    Parameters
    ----------
    source : str | Path
        Image file, image folder, or video path.
    weights : str | Path
        Path to .pt weights.
    conf, iou : float
        Detection thresholds.
    save_images : bool
        Save annotated images to outputs/predictions/.
    save_json_results : bool
        Save a JSON summary of detections.

    Returns
    -------
    list of ultralytics Results objects.
    """
    run_dir = ensure_dir(PREDICTIONS_DIR / f"run_{timestamp()}")

    detector = GasFlareDector(weights=weights, conf=conf, iou=iou)

    print(f"[INFO] Source   : {source}")
    print(f"[INFO] Conf     : {conf}  |  IoU : {iou}")
    print(f"[INFO] Save dir : {run_dir}")

    results = detector.predict(
        source=source,
        save=save_images,
        save_dir=run_dir,
    )

    # ── Build JSON summary ───────────────────────────────────────────────────
    if save_json_results:
        summary = []
        for r in results:
            boxes = r.boxes
            detections = []
            if boxes is not None and len(boxes):
                for box in boxes:
                    detections.append({
                        "class_id"  : int(box.cls.item()),
                        "class_name": detector.class_names[int(box.cls.item())],
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox_xyxy" : [round(v, 2) for v in box.xyxy[0].tolist()],
                    })
            summary.append({
                "image"     : Path(r.path).name,
                "detections": detections,
                "count"     : len(detections),
            })

        json_path = run_dir / "detections.json"
        save_json(summary, json_path)
        print(f"[INFO] JSON results saved : {json_path}")

    total = sum(len(r.boxes) if r.boxes else 0 for r in results)
    print(f"[INFO] Processed {len(results)} image(s) — {total} detection(s) total.")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Gas Flare Detector – inference")
    parser.add_argument("--source",  required=True, help="Image / folder / video path")
    parser.add_argument("--weights", default=str(BEST_WEIGHTS), help="Path to .pt weights")
    parser.add_argument("--conf",    type=float, default=CONF_THRES,  help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=IOU_THRES,   help="IoU (NMS) threshold")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated images")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        save_images=not args.no_save,
    )
