"""
Evaluation script – run the trained model on the test split and report metrics.

Usage (standalone):
    python -m src.evaluation.metrics [--split test] [--save]
"""

import argparse
from pathlib import Path

from src.models.yolo_model import GasFlareDector
from src.utils.config import (
    BEST_WEIGHTS,
    CLASS_NAMES,
    CONF_THRES,
    IOU_THRES,
    DATA_YAML,
    LOGS_DIR,
)
from src.utils.helpers import ensure_dir, save_json, save_csv, format_metrics, timestamp


def evaluate(
    weights: str | Path = BEST_WEIGHTS,
    data_yaml: str | Path = DATA_YAML,
    split: str = "test",
    conf: float = CONF_THRES,
    iou: float = IOU_THRES,
    save_results: bool = True,
) -> dict:
    """
    Evaluate the model and return a metrics dictionary.

    Parameters
    ----------
    weights   : path to .pt weights file
    data_yaml : path to dataset YAML
    split     : "train", "val", or "test"
    conf, iou : detection thresholds
    save_results : persist JSON + CSV to outputs/logs/

    Returns
    -------
    dict with keys: mAP50, mAP50_95, per_class
    """
    detector = GasFlareDector(weights=weights, conf=conf, iou=iou)

    print(f"[INFO] Evaluating on split : '{split}'")
    print(f"[INFO] Dataset YAML        : {data_yaml}")

    metrics = detector.evaluate(data_yaml=data_yaml, split=split)

    # ── Pretty print ─────────────────────────────────────────────────────────
    print(format_metrics(metrics, CLASS_NAMES))

    # ── Build result dict ────────────────────────────────────────────────────
    per_class = {
        cls: round(float(v), 4)
        for cls, v in zip(CLASS_NAMES, metrics.box.maps)
    }

    result = {
        "split"     : split,
        "weights"   : str(weights),
        "conf_thres": conf,
        "iou_thres" : iou,
        "mAP50"     : round(float(metrics.box.map50), 4),
        "mAP50_95"  : round(float(metrics.box.map),   4),
        "per_class" : per_class,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_results:
        log_dir  = ensure_dir(LOGS_DIR)
        ts       = timestamp()

        json_path = log_dir / f"eval_{split}_{ts}.json"
        save_json(result, json_path)
        print(f"[INFO] JSON report : {json_path}")

        csv_rows = [{"class": cls, "mAP50_95": v} for cls, v in per_class.items()]
        csv_rows.insert(0, {"class": "ALL", "mAP50_95": result["mAP50_95"]})
        csv_path = log_dir / f"eval_{split}_{ts}.csv"
        save_csv(csv_rows, csv_path)
        print(f"[INFO] CSV  report : {csv_path}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Gas Flare Detector – evaluation")
    parser.add_argument("--weights",   default=str(BEST_WEIGHTS), help="Path to .pt weights")
    parser.add_argument("--data",      default=str(DATA_YAML),    help="Path to data.yaml")
    parser.add_argument("--split",     default="test",            choices=["train", "val", "test"])
    parser.add_argument("--conf",      type=float, default=CONF_THRES)
    parser.add_argument("--iou",       type=float, default=IOU_THRES)
    parser.add_argument("--no-save",   action="store_true", help="Skip saving reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        weights=args.weights,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_results=not args.no_save,
    )
