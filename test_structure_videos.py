# -*- coding: utf-8 -*-
"""
Test visuel du modèle auxiliaire STRUCTURE (YOLO11-n) sur TOUTES les vidéos.

Pour chaque vidéo de data/ : annote chaque frame avec les détections du
modèle structure (mât/cheminée en épais, flamme/fumées en fin) et écrit la
vidéo complète dans outputs/structure_videos/. Imprime un bilan par vidéo
(% de frames où le mât est vu).

Usage :
    python test_structure_videos.py            # toutes les vidéos
    python test_structure_videos.py --conf 0.3
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT       = Path(__file__).resolve().parent
WEIGHTS    = ROOT / "outputs" / "models" / "flare_structure_yolo11n_v1" / "weights" / "best.pt"
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs" / "structure_videos"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

NAMES  = ["chimney", "fire smoke", "flame", "flare stack", "normal smoke"]
COLORS = {0: (255, 200, 0),    # chimney   — cyan/bleu
          1: (0, 140, 255),    # fire smoke — orange
          2: (0, 0, 255),      # flame      — rouge
          3: (0, 255, 0),      # flare stack — VERT
          4: (190, 190, 190)}  # normal smoke — gris
STRUCT = {0, 3}                # classes structure (trait épais)


def annoter_frame(frame, boxes, conf_min):
    n_stack = n_chim = 0
    for box in boxes:
        c = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ep = 3 if c in STRUCT else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[c], ep)
        cv2.putText(frame, f"{NAMES[c]} {conf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 2)
        if c == 3:
            n_stack += 1
        elif c == 0:
            n_chim += 1
    return n_stack, n_chim


def hud(frame, nom, n_stack, n_chim, frame_idx, total):
    txt = f"{nom}  |  flare stack: {n_stack}  chimney: {n_chim}  |  {frame_idx}/{total}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (8, 8), (16 + tw, 20 + th), (0, 0, 0), -1)
    couleur = (0, 255, 0) if n_stack else (0, 200, 255) if n_chim else (200, 200, 200)
    cv2.putText(frame, txt, (12, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                couleur, 2)


def main(conf=0.30):
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Poids introuvables : {WEIGHTS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    videos = sorted(p for p in DATA_DIR.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    print(f"[INFO] {len(videos)} vidéos  |  conf={conf}  |  sortie : {OUT_DIR}\n")

    bilan = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] illisible : {video.name}")
            continue
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = OUT_DIR / f"{video.stem}_structure.mp4"
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        t0 = time.time()
        frames_stack = frames_chim = idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            res = model.predict(frame, conf=0.10, iou=0.45, verbose=False)[0]
            n_stack, n_chim = annoter_frame(frame, res.boxes, conf)
            hud(frame, video.stem[:30], n_stack, n_chim, idx, tot)
            writer.write(frame)
            frames_stack += bool(n_stack)
            frames_chim  += bool(n_chim)
        cap.release()
        writer.release()

        pct_stack = frames_stack / max(idx, 1) * 100
        pct_chim  = frames_chim / max(idx, 1) * 100
        fps_proc  = idx / max(time.time() - t0, 1e-6)
        verdict = ("OK " if pct_stack >= 50 else
                   "~~ " if pct_stack >= 10 or pct_chim >= 30 else "-- ")
        bilan.append((verdict, video.name, pct_stack, pct_chim))
        print(f"  [{verdict}] {video.name:52s} mât vu {pct_stack:5.1f}% "
              f"des frames | cheminée {pct_chim:5.1f}% | {fps_proc:5.1f} FPS")

    print("\n──────── BILAN ────────")
    ok  = sum(1 for v, *_ in bilan if v == "OK ")
    mid = sum(1 for v, *_ in bilan if v == "~~ ")
    print(f"  OK (mât vu ≥50% du temps)     : {ok}/{len(bilan)}")
    print(f"  Partiel (intermittent)        : {mid}/{len(bilan)}")
    print(f"  Échec (quasi jamais vu)       : {len(bilan) - ok - mid}/{len(bilan)}")
    print(f"\n  Vidéos annotées : {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test structure sur toutes les vidéos")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="Seuil de confiance affiché (def. 0.30)")
    args = ap.parse_args()
    main(conf=args.conf)
