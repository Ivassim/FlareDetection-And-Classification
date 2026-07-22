"""
Main entry point for the Gas Flare Detection project.

Commands
--------
train    -- Train the YOLO11m model from scratch.
predict  -- Run inference on an image / folder / video.
evaluate -- Evaluate the trained model on a dataset split.

Examples
--------
  python main.py train
  python main.py predict --source path/to/image.jpg
  python main.py predict --source path/to/folder --conf 0.3 --no-save
  python main.py evaluate
  python main.py evaluate --split val
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Gas Flare Detection – YOLO11m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────────
    subparsers.add_parser("train", help="Train the model (see src/models/train.py for config)")

    # ── predict ──────────────────────────────────────────────────────────
    pred = subparsers.add_parser("predict", help="Run inference on images / video")
    pred.add_argument("--source",  required=True, help="Image file, folder, or video path")
    pred.add_argument("--weights", default=None,  help="Override default best.pt path")
    pred.add_argument("--conf",    type=float, default=None, help="Confidence threshold")
    pred.add_argument("--iou",     type=float, default=None, help="IoU (NMS) threshold")
    pred.add_argument("--no-save", action="store_true",      help="Do not save annotated images")

    # ── evaluate ──────────────────────────────────────────────────────────
    evl = subparsers.add_parser("evaluate", help="Evaluate model on a dataset split")
    evl.add_argument("--weights", default=None,   help="Override default best.pt path")
    evl.add_argument("--data",    default=None,   help="Override default data.yaml path")
    evl.add_argument("--split",   default="test", choices=["train", "val", "test"])
    evl.add_argument("--conf",    type=float, default=None)
    evl.add_argument("--iou",     type=float, default=None)
    evl.add_argument("--no-save", action="store_true", help="Skip saving reports")

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        from src.models.train_v3 import train
        train()

    elif args.command == "predict":
        from src.models.predict import predict
        from src.utils.config import BEST_WEIGHTS, CONF_THRES, IOU_THRES
        predict(
            source=args.source,
            weights=args.weights or BEST_WEIGHTS,
            conf=args.conf    or CONF_THRES,
            iou=args.iou      or IOU_THRES,
            save_images=not args.no_save,
        )

    elif args.command == "evaluate":
        from src.evaluation.metrics import evaluate
        from src.utils.config import BEST_WEIGHTS, CONF_THRES, IOU_THRES, DATA_YAML
        evaluate(
            weights=args.weights  or BEST_WEIGHTS,
            data_yaml=args.data   or DATA_YAML,
            split=args.split,
            conf=args.conf        or CONF_THRES,
            iou=args.iou          or IOU_THRES,
            save_results=not args.no_save,
        )


if __name__ == "__main__":
    main()
