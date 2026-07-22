"""
Thin wrapper around ultralytics YOLO for inference and evaluation.
"""

from pathlib import Path
from ultralytics import YOLO

from src.utils.config import (
    BEST_WEIGHTS,
    CLASS_NAMES,
    CONF_THRES,
    IOU_THRES,
    IMG_SIZE,
    DEVICE,
)


class GasFlareDector:
    """
    Wrapper around a trained YOLO model for gas flare / smoke detection.

    Parameters
    ----------
    weights : str | Path, optional
        Path to the .pt weights file.  Defaults to ``BEST_WEIGHTS``.
    device : int | str, optional
        Inference device (0 = first GPU, "cpu" = CPU).
    conf : float
        Confidence threshold for detections.
    iou : float
        IoU threshold for NMS.
    imgsz : int
        Inference image size.
    """

    def __init__(
        self,
        weights: str | Path = BEST_WEIGHTS,
        device: int | str = DEVICE,
        conf: float = CONF_THRES,
        iou: float = IOU_THRES,
        imgsz: int = IMG_SIZE,
    ):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.weights}\n"
                "Run train.py first to generate the model weights."
            )
        self.device = device
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.class_names = CLASS_NAMES

        print(f"[INFO] Loading weights : {self.weights}")
        self.model = YOLO(str(self.weights))

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, source, save: bool = False, save_dir: str | Path | None = None, **kwargs):
        """
        Run inference on *source* (image path, folder, video, URL, …).

        Returns the ultralytics Results list.
        """
        return self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            save=save,
            project=str(save_dir.parent) if save_dir else None,
            name=save_dir.name if save_dir else None,
            exist_ok=True,
            **kwargs,
        )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, data_yaml: str | Path, split: str = "test", **kwargs):
        """
        Evaluate the model on a dataset split using ultralytics val().

        Parameters
        ----------
        data_yaml : str | Path
            Path to the dataset YAML file.
        split : str
            Dataset split: "train", "val", or "test".

        Returns
        -------
        ultralytics.utils.metrics.DetMetrics
        """
        return self.model.val(
            data=str(data_yaml),
            split=split,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            **kwargs,
        )
