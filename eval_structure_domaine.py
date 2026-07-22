# -*- coding: utf-8 -*-
"""
Évaluation de l'écart de domaine du modèle STRUCTURE (YOLO11-n).

Le modèle structure est entraîné sur des panoramas lointains (Ulsan).
Ce script vérifie s'il généralise aux plans moyens/rapprochés de NOS vidéos :
  - extrait 3 frames (25 %, 50 %, 75 %) de chaque vidéo de data/
  - lance le modèle structure dessus (conf basse pour voir tout)
  - sauve les images annotées dans outputs/structure_eval/
  - imprime un verdict par vidéo (mât détecté ou non)

À lancer APRÈS l'entraînement (lancer_entrainement_structure.bat) :
    python eval_structure_domaine.py
    python eval_structure_domaine.py --conf 0.20
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT       = Path(__file__).resolve().parent
WEIGHTS    = ROOT / "outputs" / "models" / "flare_structure_yolo11n_v1" / "weights" / "best.pt"
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs" / "structure_eval"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

# Classes du modèle structure (u2_eo) — on ne s'intéresse qu'à la STRUCTURE
NAMES         = ["chimney", "fire smoke", "flame", "flare stack", "normal smoke"]
STRUCT_CLASSES = {0, 3}                      # chimney, flare stack
COLORS = {0: (255, 200, 0), 1: (0, 140, 255), 2: (0, 0, 255),
          3: (0, 255, 0), 4: (180, 180, 180)}


def annoter(frame, results, conf_min):
    """Dessine toutes les détections ; retourne le nb de structures trouvées."""
    n_struct = 0
    for box in results.boxes:
        c = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ep = 3 if c in STRUCT_CLASSES else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[c], ep)
        cv2.putText(frame, f"{NAMES[c]} {conf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[c], 2)
        if c in STRUCT_CLASSES:
            n_struct += 1
    return n_struct


def main(conf=0.25):
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {WEIGHTS}\n"
            "Lancez d'abord l'entraînement : lancer_entrainement_structure.bat")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    videos = sorted(p for p in DATA_DIR.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"[WARN] Aucune vidéo dans {DATA_DIR}")
        return

    print(f"[INFO] Modèle   : {WEIGHTS}")
    print(f"[INFO] Conf min : {conf}")
    print(f"[INFO] Vidéos   : {len(videos)}  →  sorties dans {OUT_DIR}\n")

    bilan = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] Illisible : {video.name}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_struct_video = 0
        for pct in (0.25, 0.50, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
            ret, frame = cap.read()
            if not ret:
                continue
            results = model.predict(frame, conf=0.10, iou=0.45,
                                    verbose=False)[0]
            n = annoter(frame, results, conf)
            n_struct_video += n
            out = OUT_DIR / f"{video.stem}_f{int(pct * 100)}.jpg"
            cv2.imwrite(str(out), frame)
        cap.release()

        verdict = "OK  mât détecté" if n_struct_video else "--  AUCUNE structure"
        bilan.append((video.name, n_struct_video, verdict))
        print(f"  [{verdict}]  {video.name}  ({n_struct_video} structures/3 frames)")

    n_ok = sum(1 for _, n, _ in bilan if n)
    print(f"\n[BILAN] structures détectées sur {n_ok}/{len(bilan)} vidéos.")
    print(f"[BILAN] Images annotées : {OUT_DIR}")
    if n_ok < len(bilan) * 0.6:
        print("[BILAN] Généralisation FAIBLE → prévoir d'ajouter des frames "
              "annotées de nos vidéos (plans rapprochés) avant d'activer le "
              "filtre structure en production.")
    else:
        print("[BILAN] Généralisation correcte → le filtre structure du "
              "pipeline peut être activé (il l'est par défaut dès que "
              "best.pt existe).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test écart de domaine structure")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Seuil de confiance pour compter une structure (def. 0.25)")
    args = ap.parse_args()
    main(conf=args.conf)
