# -*- coding: utf-8 -*-
"""
FlareProcessor — cœur du pipeline de surveillance torchères, réutilisable.

Encapsule le chargement des modèles (YOLO11-m + SVM 113-features) et le
traitement frame-par-frame, en réutilisant TOUTES les fonctions de
`realtime_monitor` (groupement, tracking, classification, dessins). La logique
par frame est une copie fidèle de la boucle de `run_pipeline`, afin que l'UI et
la CLI produisent des résultats identiques. `run_pipeline` n'est pas modifié.

Utilisé par l'interface graphique (app_gui.py) et par le mode batch.
"""
import time
import csv as _csv
from pathlib import Path
from datetime import datetime

import cv2

import realtime_monitor as rm   # même dossier (src/) — réutilise helpers + constantes


# ── Validation par STRUCTURE (anti-fausses-torchères type briquet) ─────────
# Modèle auxiliaire YOLO11-n entraîné sur u2_eo (chimney / flare stack…).
# S'il est présent, une flamme n'est PROMUE en nouvelle torchère que si une
# structure (mât/cheminée) la supporte géométriquement. Les torchères déjà
# labellisées ne sont jamais désactivées par ce filtre (gating de promotion
# uniquement). Sans best.pt, le pipeline est strictement identique à avant.
STRUCTURE_WEIGHTS = (Path(__file__).resolve().parents[1] / "outputs" / "models"
                     / "flare_structure_yolo11n_v1" / "weights" / "best.pt")
STRUCTURE_CLASSES = {0, 3}   # chimney, flare stack (classes du dataset u2_eo)
STRUCTURE_CONF    = 0.30     # seuil de confiance structure
STRUCTURE_EVERY   = 5        # inférence structure 1 frame sur N (cache entre-temps)
STRUCTURE_EXP_H   = 0.8      # expansion verticale vers le HAUT de la boîte structure
STRUCTURE_EXP_W   = 0.3      # expansion horizontale de chaque côté


def _structure_supporte_flamme(flame_bbox, structure_bboxes):
    """True si une boîte structure (étendue vers le haut) touche la flamme."""
    fx1, fy1, fx2, fy2 = flame_bbox
    for sx1, sy1, sx2, sy2 in structure_bboxes:
        sw, sh = sx2 - sx1, sy2 - sy1
        ex1 = sx1 - STRUCTURE_EXP_W * sw
        ex2 = sx2 + STRUCTURE_EXP_W * sw
        ey1 = sy1 - STRUCTURE_EXP_H * sh   # la flamme surmonte le mât
        if min(fx2, ex2) > max(fx1, ex1) and min(fy2, sy2) > max(fy1, ey1):
            return True
    return False


class FlareProcessor:
    """Pipeline détection + classification, pilotable frame par frame."""

    def __init__(self, model_path=None, conf=0.4, no_svm=False, structure=True):
        from ultralytics import YOLO
        self.conf = conf
        self.no_svm = no_svm
        self.model_path = str(model_path or rm.trouver_meilleur_modele())

        # Modèle structure auxiliaire (optionnel — None si pas encore entraîné)
        self.structure_model = None
        if structure and STRUCTURE_WEIGHTS.exists():
            self.structure_model = YOLO(str(STRUCTURE_WEIGHTS))
        self.mode_structure = ("Filtre structure YOLO11-n"
                               if self.structure_model is not None
                               else "Sans filtre structure")
        # Commutateur runtime : le filtre n'est appliqué que si True.
        # Éval du 12/06 : généralisation faible hors panoramas (9/20 vidéos)
        # → OFF par défaut ; la GUI l'active uniquement en mode webcam.
        self.structure_filter_on = False

        # SVM 113-features (ou fallback HSV si no_svm)
        self.svm, self.scaler, self.fn_features = rm.charger_classificateur_svm(
            forcer_hsv=no_svm)
        self.mode_classif = ("SVM 113-features" if self.svm is not None
                             else "Heuristique HSV")

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

        self.start(1920, 1080, 25.0, 0)

    # ------------------------------------------------------------------ état
    def start(self, w, h, fps_video, total_frames=0):
        """(Ré)initialise l'état pour une nouvelle source. Modèles conservés."""
        self.w, self.h = int(w), int(h)
        self.fps_video = fps_video or 25.0
        self.total_frames = int(total_frames)
        self.stats_qualite = {"bonne": 0, "moyenne": 0, "mauvaise": 0}
        self.stats_par_torchere = {}
        self.total_detections = 0
        self.stats_ambient = 0
        self.stats_inactive = 0
        self.stats_rejet_structure = 0
        self._structure_boxes = []
        self.frame_idx = 0
        self.fps_history = []
        self.tracker = rm.TrackerTorcheres(dist_max=max(self.w, self.h) // 3)

    # --------------------------------------------------------------- 1 frame
    def process_frame(self, frame):
        """
        Traite UNE frame BGR. Dessine les annotations dessus (in place).
        Retourne (frame_annotée, csv_rows, fps_current, snapshot).
        Copie fidèle de la boucle de run_pipeline (MODIF-1..26).
        """
        t0 = time.time()
        self.frame_idx += 1
        frame_idx = self.frame_idx
        timestamp_sec = frame_idx / self.fps_video
        w, h = self.w, self.h
        csv_rows = []
        frame_quality = {}   # qualité de CHAQUE torchère active sur CETTE frame (pour alertes)

        # Inférence YOLO (seuil bas + filtrage per-classe)
        results = self.model.predict(
            frame, conf=0.20, iou=0.45, verbose=False, stream=False)[0]

        # ── Pass 1 : collecter + qualifier chaque détection ──────────────
        frame_detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            confidence = float(box.conf[0])

            seuil_classe = rm.CLASS_CONF_MIN.get(cls_name, self.conf)
            if confidence < seuil_classe:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = frame[y1:y2, x1:x2]

            qualite_ind, scores = rm.analyser_qualite_combustion(
                roi, cls_name=cls_name,
                svm=self.svm, scaler=self.scaler, fn_features=self.fn_features)

            frame_detections.append({
                "cls_name": cls_name, "conf": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "qualite": qualite_ind, "scores": scores,
            })

        # ── Pass 2 : groupement + fumée ambiante ─────────────────────────
        groupes, ambient_indices = rm.grouper_torcheres(frame_detections)

        has_flame_per_group = [
            any("flare" in frame_detections[i]["cls_name"].lower() for i in g)
            for g in groupes
        ]

        # ── Filtre STRUCTURE : gating de promotion des nouvelles flammes ──
        # Une flamme sans mât/cheminée détecté dessous reste candidate
        # (label=None, Niveau 2) et ne devient jamais une torchère.
        if (self.structure_model is not None and self.structure_filter_on
                and any(has_flame_per_group)):
            if (self.frame_idx - 1) % STRUCTURE_EVERY == 0:
                s_res = self.structure_model.predict(
                    frame, conf=STRUCTURE_CONF, iou=0.45,
                    verbose=False, stream=False)[0]
                self._structure_boxes = [
                    tuple(map(int, b.xyxy[0])) for b in s_res.boxes
                    if int(b.cls[0]) in STRUCTURE_CLASSES
                ]
            for g_idx, indices in enumerate(groupes):
                if not has_flame_per_group[g_idx]:
                    continue
                # Une torchère déjà labellisée n'est pas re-filtrée :
                # assigner_labels conserve les labels existants même si
                # has_flame=False (le filtre ne bloque que la CRÉATION).
                flammes = [frame_detections[i] for i in indices
                           if "flare" in frame_detections[i]["cls_name"].lower()]
                supporte = any(
                    _structure_supporte_flamme(
                        (d["x1"], d["y1"], d["x2"], d["y2"]),
                        self._structure_boxes)
                    for d in flammes)
                if not supporte:
                    has_flame_per_group[g_idx] = False
                    self.stats_rejet_structure += 1

        labels_stables = self.tracker.assigner_labels(
            frame_detections, groupes, has_flame_per_group)
        torcheres_info = []

        # Fumée ambiante
        for amb_idx in ambient_indices:
            det = frame_detections[amb_idx]
            self.stats_ambient += 1
            rm.dessiner_detection(
                frame, det["x1"], det["y1"], det["x2"], det["y2"],
                det["cls_name"], det["conf"], "ambient", torchere_label="")
            csv_rows.append([
                frame_idx, f"{timestamp_sec:.3f}", "ambient",
                det["cls_name"], f"{det['conf']:.4f}", "ambient_smoke",
                det["x1"], det["y1"], det["x2"], det["y2"],
                det["scores"].get("dark_ratio", ""),
                det["scores"].get("flame_ratio", ""),
                det["scores"].get("smoke_ratio", ""),
                det["scores"].get("mean_intensity", ""),
                det["scores"].get("source", "n/a"),
            ])

        # ── Traiter chaque groupe ────────────────────────────────────────
        for g_idx, indices in enumerate(groupes):
            label_full = labels_stables[g_idx]
            if label_full is None:          # candidat sans flamme (Niveau 2)
                continue
            label_court = label_full.split()[-1]

            has_flame = any(
                "flare" in frame_detections[i]["cls_name"].lower()
                for i in indices)
            state = self.tracker.get_state(label_court)
            is_active = state.update(has_flame)

            if label_full not in self.stats_par_torchere:
                self.stats_par_torchere[label_full] = {
                    "bonne": 0, "moyenne": 0, "mauvaise": 0, "inactive": 0}

            if not is_active:
                torcheres_info.append({
                    "label": label_full, "qualite": "inactive",
                    "n_detections": len(indices)})
                for det_idx in indices:
                    det = frame_detections[det_idx]
                    self.stats_inactive += 1
                    self.stats_par_torchere[label_full]["inactive"] += 1
                    csv_rows.append([
                        frame_idx, f"{timestamp_sec:.3f}",
                        label_full, det["cls_name"], f"{det['conf']:.4f}",
                        "inactive",
                        det["x1"], det["y1"], det["x2"], det["y2"],
                        det["scores"].get("dark_ratio", ""),
                        det["scores"].get("flame_ratio", ""),
                        det["scores"].get("smoke_ratio", ""),
                        det["scores"].get("mean_intensity", ""),
                        det["scores"].get("source", "n/a"),
                    ])
                continue

            # Torchère active : classification qualité du groupe
            qualite_groupe = rm.determiner_qualite_torchere(
                frame_detections, indices)
            frame_quality[label_full] = qualite_groupe
            torcheres_info.append({
                "label": label_full, "qualite": qualite_groupe,
                "n_detections": len(indices)})

            primary_idx = next(
                (i for i in indices
                 if "flare" in frame_detections[i]["cls_name"].lower()),
                indices[0])

            for det_idx in indices:
                det = frame_detections[det_idx]
                self.total_detections += 1
                self.stats_qualite[qualite_groupe] += 1
                self.stats_par_torchere[label_full][qualite_groupe] += 1
                label_drawn = label_full if det_idx == primary_idx else ""
                rm.dessiner_detection(
                    frame, det["x1"], det["y1"], det["x2"], det["y2"],
                    det["cls_name"], det["conf"], qualite_groupe,
                    torchere_label=label_drawn)
                csv_rows.append([
                    frame_idx, f"{timestamp_sec:.3f}",
                    label_full, det["cls_name"], f"{det['conf']:.4f}",
                    qualite_groupe,
                    det["x1"], det["y1"], det["x2"], det["y2"],
                    det["scores"].get("dark_ratio", ""),
                    det["scores"].get("flame_ratio", ""),
                    det["scores"].get("smoke_ratio", ""),
                    det["scores"].get("mean_intensity", ""),
                    det["scores"].get("source", "n/a"),
                ])

        # ── Panneau + HUD ────────────────────────────────────────────────
        if len(torcheres_info) > 1:
            rm.dessiner_panel_torcheres(frame, torcheres_info)

        fps_current = 1.0 / max(time.time() - t0, 1e-6)
        self.fps_history.append(fps_current)
        rm.dessiner_hud(frame, fps_current, frame_idx, self.total_frames,
                        self.stats_qualite, self.stats_ambient,
                        self.stats_inactive)

        snap = self.snapshot()
        snap["current_quality"] = frame_quality   # {label: bonne|moyenne|mauvaise} sur cette frame
        return frame, csv_rows, fps_current, snap

    # ----------------------------------------------------------- accesseurs
    def snapshot(self):
        return {
            "frame_idx": self.frame_idx,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "stats_qualite": dict(self.stats_qualite),
            "stats_ambient": self.stats_ambient,
            "stats_inactive": self.stats_inactive,
            "stats_rejet_structure": self.stats_rejet_structure,
            "structure_active": self.structure_model is not None,
            "stats_par_torchere": {k: dict(v)
                                   for k, v in self.stats_par_torchere.items()},
        }

    def retained_torcheres(self):
        """Torchères retenues (≥ MIN_DISPLAY_ACTIVE détections actives)."""
        out = {}
        for name, tq in self.stats_par_torchere.items():
            actif = sum(c for q, c in tq.items() if q != "inactive")
            if actif >= rm.MIN_DISPLAY_ACTIVE:
                out[name] = tq
        return out

    def fps_moyen(self):
        return sum(self.fps_history) / max(len(self.fps_history), 1)

    def recap_frame(self):
        return rm.dessiner_recap_final(
            self.w, self.h, self.stats_par_torchere, self.stats_qualite,
            self.total_detections, self.frame_idx, self.fps_moyen(),
            self.stats_ambient, self.stats_inactive)

    def verdict(self):
        """Qualité dominante globale (sur détections actives), ou None."""
        sq = self.stats_qualite
        tot = sum(sq.values())
        if tot == 0:
            return None, 0.0
        dom = max(sq, key=sq.get)
        return dom, sq[dom] / tot * 100

    # ----------------------------------------------------- batch / fichier
    def process_video_file(self, source, output_dir=None, write_video=True,
                           write_csv=True, add_recap=True, progress_cb=None):
        """
        Traite une vidéo entière (mode batch). Écrit MP4 annoté + CSV si demandé.
        progress_cb(snapshot) appelé périodiquement. Retourne un dict de résultats.
        """
        src = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir : {source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start(w, h, fps, total)

        writer = csv_file = csv_writer = None
        out_video = out_csv = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(str(source)).stem
            if write_video:
                out_video = output_dir / f"{stem}_annot_{ts}.mp4"
                writer = cv2.VideoWriter(
                    str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if write_csv:
                out_csv = output_dir / f"{stem}_log_{ts}.csv"
                csv_file = open(out_csv, "w", newline="", encoding="utf-8")
                csv_writer = _csv.writer(csv_file)
                csv_writer.writerow([
                    "frame_idx", "timestamp_sec", "torchere", "class_name",
                    "confidence", "qualite", "x1", "y1", "x2", "y2",
                    "dark_ratio", "flame_ratio", "smoke_ratio",
                    "mean_intensity", "classif_source"])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, rows, _fps, snap = self.process_frame(frame)
                if writer is not None:
                    writer.write(frame)
                if csv_writer is not None:
                    csv_writer.writerows(rows)
                if progress_cb and self.frame_idx % 10 == 0:
                    progress_cb(snap)
            if add_recap and writer is not None and self.stats_par_torchere:
                recap = self.recap_frame()
                for _ in range(int(fps * 3)):
                    writer.write(recap)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()

        dom, pct = self.verdict()
        sq, tot = self.stats_qualite, max(self.total_detections, 1)
        return {
            "source": str(source),
            "frames": self.frame_idx,
            "fps_moyen": self.fps_moyen(),
            "torcheres_retenues": len(self.retained_torcheres()),
            "torcheres_detectees": len(self.stats_par_torchere),
            "pct_bonne": sq["bonne"] / tot * 100,
            "pct_moyenne": sq["moyenne"] / tot * 100,
            "pct_mauvaise": sq["mauvaise"] / tot * 100,
            "verdict": dom, "verdict_pct": pct,
            "video": str(out_video) if out_video else None,
            "csv": str(out_csv) if out_csv else None,
        }
