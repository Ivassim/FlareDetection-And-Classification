# -*- coding: utf-8 -*-
"""
Interface de surveillance des torchères — SONATRACH × USTHB (PFE).

Appli bureau PySide6 :
  • Temps réel : tester une vidéo (ou la webcam) avec flux annoté live + stats
  • Batch      : traiter plusieurs vidéos et comparer les résultats

Lancer :  .venv\\Scripts\\python.exe app_gui.py
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget, QProgressBar,
    QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView, QFrame,
    QSizePolicy, QAbstractItemView, QListWidget, QListWidgetItem,
)

from flare_processor import FlareProcessor
import realtime_monitor as rm

# ── Thème ────────────────────────────────────────────────────────────────
BG = "#15161a"; PANEL = "#1e2026"; INK = "#eaeaea"; MUT = "#9aa0aa"
BLUE = "#2433A6"; GREEN = "#28b85a"; ORANGE = "#e9920a"; RED = "#e23b3b"
QCOLOR = {"bonne": GREEN, "moyenne": ORANGE, "mauvaise": RED, "inactive": "#888"}
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
ALERT_FRAMES = 8        # frames consécutives de combustion mauvaise avant alerte (anti-faux positif)
ALERT_CLEAR_FRAMES = 45 # frames consécutives NON-mauvaises avant de lever l'alerte (hystérésis
                        # de sortie : évite le clignotement "rétablie/mauvaise" sur le scintillement)
ALERT_RED = "#e23b3b"; ALERT_DARK = "#9c1f1f"


# ═════════════════════════════════════════════════════════ Workers (threads)
class ModelLoader(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.ready.emit(FlareProcessor())
        except Exception as e:      # noqa
            self.failed.emit(repr(e))


class RealtimeWorker(QThread):
    frame_ready = Signal(QImage, dict, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, processor, source):
        super().__init__()
        self.processor = processor
        self.source = source
        self._paused = False
        self._stop = False

    def pause(self, p): self._paused = p
    def stop(self): self._stop = True

    def run(self):
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error.emit(f"Impossible d'ouvrir la source : {self.source}")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.processor.start(w, h, fps, total)
        try:
            while not self._stop:
                if self._paused:
                    self.msleep(40); continue
                ret, frame = cap.read()
                if not ret:
                    break
                frame, _rows, fps_cur, snap = self.processor.process_frame(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hh, ww, _ = rgb.shape
                qimg = QImage(rgb.data, ww, hh, 3 * ww, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg, snap, fps_cur)
        except Exception as e:      # noqa
            self.error.emit(repr(e))
        finally:
            cap.release()
        self.finished.emit(self.processor.snapshot())


class BatchWorker(QThread):
    row_ready = Signal(int, dict)
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, processor, videos, out_dir):
        super().__init__()
        self.processor = processor
        self.videos = videos
        self.out_dir = out_dir
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        for i, v in enumerate(self.videos):
            if self._stop:
                break
            self.progress.emit(i, f"Traitement {i+1}/{len(self.videos)} : {Path(v).name}")
            try:
                res = self.processor.process_video_file(
                    v, output_dir=self.out_dir, write_video=True,
                    write_csv=True, add_recap=True)
                self.row_ready.emit(i, res)
            except Exception as e:  # noqa
                self.row_ready.emit(i, {"source": v, "error": repr(e)})
        self.finished.emit()


# ═══════════════════════════════════════════════════════════ Widgets stats
class QualityBar(QWidget):
    """Une ligne : libellé + barre colorée + compteur."""
    def __init__(self, label, color):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 2, 0, 2)
        self.name = QLabel(label); self.name.setFixedWidth(86)
        self.name.setStyleSheet(f"color:{INK};font-weight:600;")
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False)
        self.bar.setFixedHeight(16)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:#2a2d35;border:none;border-radius:8px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:8px;}}")
        self.val = QLabel("0 (0.0%)"); self.val.setFixedWidth(110)
        self.val.setStyleSheet(f"color:{MUT};")
        lay.addWidget(self.name); lay.addWidget(self.bar, 1); lay.addWidget(self.val)

    def set(self, count, total):
        pct = count / max(total, 1) * 100
        self.bar.setValue(int(pct)); self.val.setText(f"{count} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════ Fenêtre
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SONATRACH — Surveillance Torchères · PFE USTHB")
        self.resize(1280, 760)
        self.processor = None
        self.rt_worker = None
        self.batch_worker = None
        self.batch_idx = 0
        # État des alertes "combustion mauvaise"
        self.alert_counters = {}        # label -> nb frames mauvaise consécutives
        self.recover_counters = {}      # label -> nb frames NON-mauvaises consécutives (hystérésis sortie)
        self.active_alerts = set()      # labels actuellement en alerte
        self._pulse = False
        self.alert_timer = QTimer(self)
        self.alert_timer.timeout.connect(self._pulse_banner)

        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};color:{INK};
                font-family:'Segoe UI';font-size:13px;}}
            QPushButton{{background:{PANEL};border:1px solid #33363f;
                border-radius:6px;padding:7px 14px;color:{INK};}}
            QPushButton:hover{{border-color:{BLUE};}}
            QPushButton:disabled{{color:#5a5e66;border-color:#26282e;}}
            QComboBox{{background:{PANEL};border:1px solid #33363f;
                border-radius:6px;padding:6px;color:{INK};}}
            QTabWidget::pane{{border:1px solid #2a2d35;}}
            QTabBar::tab{{background:{PANEL};padding:9px 18px;color:{MUT};
                border-top-left-radius:6px;border-top-right-radius:6px;}}
            QTabBar::tab:selected{{background:{BLUE};color:white;}}
            QTableWidget{{background:{PANEL};gridline-color:#2a2d35;
                border:1px solid #2a2d35;}}
            QHeaderView::section{{background:#23262e;color:{MUT};
                padding:6px;border:none;}}
        """)

        tabs = QTabWidget()
        tabs.addTab(self._build_realtime_tab(), "  Temps réel  ")
        tabs.addTab(self._build_batch_tab(), "  Batch  ")
        self.setCentralWidget(tabs)

        self.status = self.statusBar()
        self.status.setStyleSheet(f"color:{MUT};")
        self._set_status("Chargement du modèle YOLO11-m + SVM 113-features…")
        self._set_controls_enabled(False)

        self.loader = ModelLoader()
        self.loader.ready.connect(self._on_model_ready)
        self.loader.failed.connect(lambda e: self._set_status(f"Échec chargement : {e}"))
        self.loader.start()

    # ───────────────────────────────── onglet Temps réel
    def _build_realtime_tab(self):
        w = QWidget(); root = QVBoxLayout(w)

        bar = QHBoxLayout()
        self.combo = QComboBox(); self.combo.setMinimumWidth(280)
        self._refresh_video_combo()
        self.btn_open = QPushButton("Ouvrir…")
        self.btn_cam = QPushButton("Webcam")
        self.btn_play = QPushButton("▶  Démarrer")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_open.clicked.connect(self._open_file)
        self.btn_cam.clicked.connect(self._start_webcam)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(self._stop_rt)
        for b in (QLabel("Vidéo :"), self.combo, self.btn_open, self.btn_cam):
            bar.addWidget(b)
        bar.addStretch(1); bar.addWidget(self.btn_play); bar.addWidget(self.btn_stop)
        root.addLayout(bar)

        self.alert_banner = QLabel("")
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setVisible(False)
        root.addWidget(self.alert_banner)

        body = QHBoxLayout()
        self.video = QLabel("Sélectionne une vidéo puis « Démarrer »")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(720, 460)
        self.video.setStyleSheet(f"background:#0c0d10;border:1px solid #2a2d35;color:{MUT};")
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self.video, 3)

        panel = QFrame(); panel.setFixedWidth(330)
        panel.setStyleSheet(f"background:{PANEL};border:1px solid #2a2d35;border-radius:8px;")
        pl = QVBoxLayout(panel)
        self.lbl_fps = QLabel("FPS : —    Frame : —")
        self.lbl_fps.setStyleSheet(f"color:{INK};font-weight:600;font-size:14px;")
        pl.addWidget(self.lbl_fps)
        pl.addWidget(self._sep())
        cap = QLabel("Distribution qualité"); cap.setStyleSheet(f"color:{MUT};")
        pl.addWidget(cap)
        self.qbars = {q: QualityBar(q.capitalize(), QCOLOR[q])
                      for q in ("bonne", "moyenne", "mauvaise")}
        for q in ("bonne", "moyenne", "mauvaise"):
            pl.addWidget(self.qbars[q])
        pl.addWidget(self._sep())
        cap2 = QLabel("Torchères retenues"); cap2.setStyleSheet(f"color:{MUT};")
        pl.addWidget(cap2)
        self.t_table = QTableWidget(0, 3)
        self.t_table.setHorizontalHeaderLabels(["Torchère", "Verdict", "%"])
        self.t_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.t_table.verticalHeader().setVisible(False)
        self.t_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pl.addWidget(self.t_table, 1)
        pl.addWidget(self._sep())
        cap3 = QLabel("Journal d'alertes"); cap3.setStyleSheet(f"color:{MUT};")
        pl.addWidget(cap3)
        self.alert_log = QListWidget()
        self.alert_log.setFixedHeight(120)
        self.alert_log.setStyleSheet(
            f"background:#15161a;border:1px solid #2a2d35;border-radius:6px;font-size:12px;")
        pl.addWidget(self.alert_log)
        body.addWidget(panel)
        root.addLayout(body, 1)
        return w

    # ───────────────────────────────── onglet Batch
    def _build_batch_tab(self):
        w = QWidget(); root = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.btn_add = QPushButton("Ajouter des vidéos…")
        self.btn_loaddata = QPushButton("Charger data/")
        self.btn_clear = QPushButton("Vider")
        self.btn_run_batch = QPushButton("▶  Lancer le batch")
        self.btn_add.clicked.connect(self._batch_add)
        self.btn_loaddata.clicked.connect(self._batch_load_data)
        self.btn_clear.clicked.connect(lambda: (self.batch_videos.clear(), self._batch_refresh()))
        self.btn_run_batch.clicked.connect(self._run_batch)
        bar.addWidget(self.btn_add); bar.addWidget(self.btn_loaddata)
        bar.addWidget(self.btn_clear); bar.addStretch(1); bar.addWidget(self.btn_run_batch)
        root.addLayout(bar)

        self.batch_progress = QProgressBar(); self.batch_progress.setTextVisible(True)
        root.addWidget(self.batch_progress)

        self.batch_videos = []
        self.b_table = QTableWidget(0, 7)
        self.b_table.setHorizontalHeaderLabels(
            ["Vidéo", "Torchères", "Bonne %", "Moyenne %", "Mauvaise %", "Verdict", "FPS"])
        self.b_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.b_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.b_table.verticalHeader().setVisible(False)
        self.b_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.b_table, 1)
        return w

    # ───────────────────────────────── helpers UI
    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color:#2a2d35;"); return f

    def _refresh_video_combo(self):
        self.combo.clear()
        d = ROOT / "data"
        vids = sorted([p.name for p in d.glob("*") if p.suffix.lower() in VIDEO_EXT]) if d.exists() else []
        self.combo.addItems(vids or ["(aucune vidéo dans data/)"])

    def _set_status(self, msg): self.status.showMessage(msg)

    def _set_controls_enabled(self, on):
        for b in (self.btn_play, self.btn_cam, self.btn_open,
                  self.btn_run_batch, self.btn_add, self.btn_loaddata):
            b.setEnabled(on)

    def _on_model_ready(self, processor):
        self.processor = processor
        self._set_controls_enabled(True)
        self._set_status(f"Modèle prêt — {processor.mode_classif}. Prêt à analyser.")

    # ───────────────────────────────── Temps réel : actions
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une vidéo", str(ROOT / "data"),
            "Vidéos (*.mp4 *.mov *.avi *.mkv *.m4v)")
        if path:
            self._start_source(path)

    def _start_webcam(self):
        self._start_source("0")

    def _toggle_play(self):
        if self.rt_worker and self.rt_worker.isRunning():
            paused = self.btn_play.text().startswith("▶")
            self.rt_worker.pause(paused)
            self.btn_play.setText("⏸  Pause" if paused else "▶  Reprendre")
        else:
            name = self.combo.currentText()
            p = ROOT / "data" / name
            if p.exists():
                self._start_source(str(p))
            else:
                self._set_status("Aucune vidéo valide sélectionnée.")

    def _start_source(self, source):
        if self.processor is None:
            return
        self._stop_rt()
        self._reset_alerts(clear_log=True)
        # Filtre structure (anti-fausses-torchères) : webcam uniquement —
        # sur les vidéos industrielles l'écart de domaine le rend trop strict.
        est_webcam = str(source).isdigit()
        self.processor.structure_filter_on = (
            est_webcam and self.processor.structure_model is not None)
        self.rt_worker = RealtimeWorker(self.processor, source)
        self.rt_worker.frame_ready.connect(self._on_frame)
        self.rt_worker.finished.connect(self._on_rt_finished)
        self.rt_worker.error.connect(self._set_status)
        self.rt_worker.start()
        self.btn_play.setText("⏸  Pause")
        suffixe = ("  [filtre structure actif]"
                   if self.processor.structure_filter_on else "")
        self._set_status(f"Analyse en cours : {source}{suffixe}")

    def _stop_rt(self):
        if self.rt_worker and self.rt_worker.isRunning():
            self.rt_worker.stop(); self.rt_worker.wait(2000)
        self.rt_worker = None
        self.btn_play.setText("▶  Démarrer")
        self._reset_alerts(clear_log=False)

    def _on_frame(self, qimg, snap, fps):
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        tot = max(snap["total_detections"], 1)
        for q in ("bonne", "moyenne", "mauvaise"):
            self.qbars[q].set(snap["stats_qualite"][q], tot)
        tf = snap["total_frames"]
        prog = f"{snap['frame_idx']}/{tf}" if tf > 0 else str(snap["frame_idx"])
        self.lbl_fps.setText(f"FPS : {fps:4.1f}    Frame : {prog}")
        self._update_torchere_table(snap)
        self._process_alerts(snap.get("current_quality", {}))

    def _update_torchere_table(self, snap):
        rows = []
        for name, tq in snap["stats_par_torchere"].items():
            actif = sum(c for k, c in tq.items() if k != "inactive")
            if actif < rm.MIN_DISPLAY_ACTIVE:
                continue
            dom = max(("bonne", "moyenne", "mauvaise"), key=lambda q: tq.get(q, 0))
            pct = tq.get(dom, 0) / max(actif, 1) * 100
            rows.append((name, dom, pct))
        rows.sort(key=lambda r: r[0])
        self.t_table.setRowCount(len(rows))
        for i, (name, dom, pct) in enumerate(rows):
            self.t_table.setItem(i, 0, QTableWidgetItem(name))
            it = QTableWidgetItem(dom.upper()); it.setForeground(QColor(QCOLOR[dom]))
            self.t_table.setItem(i, 1, it)
            self.t_table.setItem(i, 2, QTableWidgetItem(f"{pct:.0f}%"))

    # ───────────────────────────────── Alertes combustion mauvaise
    def _process_alerts(self, current_quality):
        bad_now = {lbl for lbl, q in current_quality.items() if q == "mauvaise"}
        newly, cleared = [], []
        suivis = (set(self.alert_counters) | set(self.recover_counters)
                  | self.active_alerts | bad_now)
        for lbl in suivis:
            if lbl in bad_now:
                # Mauvaise cette frame : on (ré)arme, on annule la récupération en cours.
                self.alert_counters[lbl] = self.alert_counters.get(lbl, 0) + 1
                self.recover_counters[lbl] = 0
                if (self.alert_counters[lbl] >= ALERT_FRAMES
                        and lbl not in self.active_alerts):
                    self.active_alerts.add(lbl); newly.append(lbl)
            else:
                # Pas mauvaise cette frame : on coupe le compteur d'activation.
                self.alert_counters[lbl] = 0
                if lbl in self.active_alerts:
                    # Hystérésis de sortie : l'alerte ne se lève qu'après
                    # ALERT_CLEAR_FRAMES frames NON-mauvaises consécutives.
                    self.recover_counters[lbl] = self.recover_counters.get(lbl, 0) + 1
                    if self.recover_counters[lbl] >= ALERT_CLEAR_FRAMES:
                        self.active_alerts.discard(lbl)
                        self.alert_counters.pop(lbl, None)
                        self.recover_counters.pop(lbl, None)
                        cleared.append(lbl)
                else:
                    # Pas (ou plus) en alerte et plus mauvaise → on arrête de suivre.
                    self.alert_counters.pop(lbl, None)
                    self.recover_counters.pop(lbl, None)
        for lbl in sorted(newly):
            self._log_alert(f"⚠  {lbl} — COMBUSTION MAUVAISE", ALERT_RED)
        for lbl in sorted(cleared):
            self._log_alert(f"{lbl} — combustion rétablie", GREEN)
        self._update_alert_banner()

    def _update_alert_banner(self):
        if self.active_alerts:
            names = ", ".join(sorted(self.active_alerts))
            self.alert_banner.setText(f"⚠   ALERTE — Combustion MAUVAISE détectée : {names}")
            self.alert_banner.setVisible(True)
            if not self.alert_timer.isActive():
                self.alert_timer.start(450)
        else:
            self.alert_banner.setVisible(False)
            self.alert_timer.stop(); self._pulse = False

    def _pulse_banner(self):
        self._pulse = not self._pulse
        c = ALERT_RED if self._pulse else ALERT_DARK
        self.alert_banner.setStyleSheet(
            f"background:{c};color:white;font-weight:700;font-size:15px;"
            f"padding:9px;border-radius:6px;")

    def _log_alert(self, msg, color):
        it = QListWidgetItem(f"[{datetime.now():%H:%M:%S}]  {msg}")
        it.setForeground(QColor(color))
        self.alert_log.insertItem(0, it)

    def _reset_alerts(self, clear_log=False):
        self.alert_counters.clear(); self.recover_counters.clear()
        self.active_alerts.clear()
        self.alert_timer.stop(); self._pulse = False
        self.alert_banner.setVisible(False)
        if clear_log:
            self.alert_log.clear()

    def _on_rt_finished(self, snap):
        self.btn_play.setText("▶  Démarrer")
        dom, pct = self.processor.verdict() if self.processor else (None, 0)
        n = len(self.processor.retained_torcheres()) if self.processor else 0
        v = f" — verdict global : {dom.upper()} ({pct:.0f}%)" if dom else ""
        self._set_status(f"Terminé : {snap['frame_idx']} frames · {n} torchère(s) retenue(s){v}")
        self._reset_alerts(clear_log=False)

    # ───────────────────────────────── Batch : actions
    def _batch_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Ajouter des vidéos", str(ROOT / "data"),
            "Vidéos (*.mp4 *.mov *.avi *.mkv *.m4v)")
        for p in paths:
            if p not in self.batch_videos:
                self.batch_videos.append(p)
        self._batch_refresh()

    def _batch_load_data(self):
        d = ROOT / "data"
        if d.exists():
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in VIDEO_EXT and str(p) not in self.batch_videos:
                    self.batch_videos.append(str(p))
        self._batch_refresh()

    def _batch_refresh(self):
        self.b_table.setRowCount(len(self.batch_videos))
        for i, v in enumerate(self.batch_videos):
            self.b_table.setItem(i, 0, QTableWidgetItem(Path(v).name))
            for c in range(1, 7):
                self.b_table.setItem(i, c, QTableWidgetItem("—"))

    def _run_batch(self):
        if self.processor is None or not self.batch_videos:
            self._set_status("Ajoute au moins une vidéo au batch."); return
        self.btn_run_batch.setEnabled(False)
        self.processor.structure_filter_on = False   # jamais en batch
        self.batch_progress.setRange(0, len(self.batch_videos))
        out = ROOT / "outputs" / "gui_batch"
        self.batch_worker = BatchWorker(self.processor, list(self.batch_videos), out)
        self.batch_worker.row_ready.connect(self._on_batch_row)
        self.batch_worker.progress.connect(lambda i, m: (self.batch_progress.setValue(i), self._set_status(m)))
        self.batch_worker.finished.connect(self._on_batch_done)
        self.batch_worker.error.connect(self._set_status)
        self.batch_worker.start()

    def _on_batch_row(self, i, res):
        def cell(c, t, color=None):
            it = QTableWidgetItem(t)
            if color: it.setForeground(QColor(color))
            self.b_table.setItem(i, c, it)
        if "error" in res:
            cell(1, "ERREUR", RED)
            self._set_status(f"Erreur sur {Path(res['source']).name} : {res['error']}")
            return
        cell(1, str(res["torcheres_retenues"]))
        cell(2, f"{res['pct_bonne']:.1f}")
        cell(3, f"{res['pct_moyenne']:.1f}")
        cell(4, f"{res['pct_mauvaise']:.1f}")
        v = res["verdict"]
        cell(5, (v.upper() if v else "—"), QCOLOR.get(v))
        cell(6, f"{res['fps_moyen']:.1f}")
        self.batch_progress.setValue(i + 1)

    def _on_batch_done(self):
        self.btn_run_batch.setEnabled(True)
        self.batch_progress.setValue(self.batch_progress.maximum())
        self._set_status(f"Batch terminé — sorties dans outputs/gui_batch/")

    def closeEvent(self, e):
        self._stop_rt()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop(); self.batch_worker.wait(3000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
