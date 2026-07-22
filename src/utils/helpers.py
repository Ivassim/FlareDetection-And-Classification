"""
Utility helpers: visualization, file I/O, and result formatting.
"""

import json
import csv
from pathlib import Path
from datetime import datetime


# ── I/O ──────────────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: list[dict], path: Path) -> None:
    """Save a list of dicts to a CSV file."""
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ── Formatting ───────────────────────────────────────────────────────────────

def timestamp() -> str:
    """Return a sortable timestamp string: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(metrics: dict, class_names: list[str]) -> str:
    """
    Pretty-print a metrics dict returned by ultralytics val().
    Expected keys: box.map50, box.map, box.maps (per-class mAP50-95).
    """
    lines = ["", "=" * 52, "  Evaluation Results", "=" * 52]

    map50    = metrics.box.map50
    map5095  = metrics.box.map
    lines.append(f"  {'mAP@0.50':<24}  {map50:.4f}")
    lines.append(f"  {'mAP@0.50:0.95':<24}  {map5095:.4f}")
    lines.append("-" * 52)
    lines.append(f"  {'Class':<22}  {'mAP50-95':>10}")
    lines.append("-" * 52)

    per_class = metrics.box.maps          # ndarray, one value per class
    for cls, val in zip(class_names, per_class):
        lines.append(f"  {cls:<22}  {val:>10.4f}")

    lines.append("=" * 52)
    return "\n".join(lines)


def get_image_paths(source: str | Path, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> list[Path]:
    """
    Collect image file paths from a file or directory.
    Returns a sorted list of Path objects.
    """
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        paths = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in extensions
        )
        return paths
    raise FileNotFoundError(f"Source not found: {source}")
