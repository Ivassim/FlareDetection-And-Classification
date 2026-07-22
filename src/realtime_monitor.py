"""
Pipeline de surveillance temps réel des torchères industrielles.

Architecture à deux étages :
    Étage 1 — YOLO11-m : Détection des objets (flammes, fumée)
    Étage 2 — Classificateur qualité de combustion (bonne/moyenne/mauvaise)
              → SVM 113-features si disponible, sinon heuristique HSV

Correctifs v2 :
    MODIF-1  FlareState : validation temporelle hystérésis double
    MODIF-2  Règle flamme obligatoire pour torchère "active"
    MODIF-3  Seuil distance fumée→torchère (DIST_MAX_SMOKE=300 px)
    MODIF-4  Fallback sans flamme : groupes séparés (pas de fusion abusive)
    MODIF-5  Oubli des torchères fantômes après TRACKER_MAX_ABSENT frames
    MODIF-6  Labels robustes au-delà de 26 torchères (AA, AB, ...)
    MODIF-7  SVM 113-features intégré (fallback HSV si modèle absent)
    MODIF-8  determiner_qualite_torchere utilise det["qualite"] (SVM/HSV)
    MODIF-9  writer.release() déplacé dans le bloc finally
    MODIF-10 fps_moyen calculé une seule fois (doublon supprimé)
    MODIF-11 Correction couleur "moyenne" : (0,165,255) orange BGR
    MODIF-12 Clip barre qualité pour éviter dépassement bas de frame

Correctifs v3 (cette version) :
    MODIF-13 Seuils de confiance par classe (CLASS_CONF_MIN) :
             *-Flare = 0.28  (capter petites flammes distantes)
             *-Smoke = 0.42-0.45 (rejeter fumées fantômes)
             Remplace le seuil global unique conf=0.4 qui ne pouvait
             satisfaire simultanément les deux classes.
    MODIF-14 model.predict(conf=0.20) bas, filtrage précis post-process.
    MODIF-15 Paramètres temporels assouplis :
             DEACTIVATION_FRAMES : 15 → 45 (tolère 1.5s sans flamme)
             TRACKER_MAX_ABSENT  : 30 → 90 (garde l'entrée 3s)
    MODIF-16 Archive de ré-identification dans TrackerTorcheres :
             Les torchères "perdues" sont déplacées dans une archive
             pendant 10s. Quand une nouvelle détection apparaît à la
             même position, elle récupère son ancien label au lieu
             d'en créer un nouveau → labels stables (plus de B→C).
    MODIF-17 État FlareState préservé partiellement à la ré-ID :
             une torchère active reste active après reconnaissance.
    MODIF-18 Label "Torchere X" dessiné une seule fois par groupe :
             affiché sur la détection PRIMAIRE (flamme prioritaire,
             sinon première détection). Les autres bboxes du même
             groupe affichent seulement classe + conf + qualité, sans
             le préfixe "Torchere X". Élimine l'illusion de "deux
             Torchère B" quand un même groupe a flamme + fumées.
    MODIF-19 Filet de sécurité : détection et correction des doublons
             de label dans assigner_labels (cas théoriquement impossible
             mais protection en profondeur).

Fonctionnalités :
    - Lecture frame-by-frame avec OpenCV
    - Inférence YOLO sur GPU (si disponible)
    - Overlay visuel : bboxes colorées, labels, FPS
    - Sauvegarde vidéo annotée (.mp4)
    - Log CSV horodaté (détections actives + fumée ambiante + inactives)

Usage :
    python src/realtime_monitor.py --source data/test_flare.mp4
    python src/realtime_monitor.py --source data/test_flare.mp4 \\
        --model outputs/models/gas_flare_yolo11m_v1/weights/best.pt
    python src/realtime_monitor.py --source 0            # webcam
    python src/realtime_monitor.py --source vid.mp4 --no-svm  # forcer HSV

Auteur : PFE Sonatrach — Surveillance Intelligente des Torchères
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Ajouter le répertoire racine au path pour les imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Paramètres de validation temporelle ──────────────────────────────────
# MODIF-1 : constantes pour FlareState
# MODIF-15 : valeurs assouplies pour réduire le flickering actif/inactif

ACTIVATION_FRAMES   = 5     # frames consécutives AVEC flamme → torchère "active"
DEACTIVATION_FRAMES = 45    # MODIF-15 : 15→45, tolère 1.5s sans flamme (occlusions, scintillement)
TRACKER_MAX_ABSENT  = 90    # MODIF-15 : 30→90, garde l'entrée active 3s avant archivage
DIST_MAX_SMOKE      = 300   # MODIF-3 : distance max (px) fumée→torchère
MIN_DISPLAY_ACTIVE  = 10    # MODIF-26 : détections actives mini pour qu'une torchère soit
                            # "retenue". En dessous = transitoire/fantôme, non affichée (console + récap).

# ── Re-identification / archive du tracker ───────────────────────────────
# MODIF-16 : permet de retrouver le label d'une torchère brièvement perdue

TRACKER_ARCHIVE_MAX_AGE  = 300  # frames (~10s @ 30fps) avant oubli définitif
TRACKER_REID_DIST_FACTOR = 1.5  # rayon de recherche archive = dist_max × ce facteur

# ── Seuils de confiance par classe YOLO ──────────────────────────────────
# MODIF-13 : remplace le seuil global unique qui ne pouvait satisfaire
# simultanément les *-Flare (besoin d'un seuil bas) et *-Smoke (haut).
#
# *-Flare bas (0.28)  : capter les petites flammes distantes / scintillantes
# *-Smoke haut (0.42+): rejeter les faux positifs (ciel, vapeur, nuages)
# Light-Smoke encore plus haut : le plus prone à confusion avec ciel clair.
# Le seuil --conf CLI sert de fallback pour les classes non listées.

CLASS_CONF_MIN = {
    "Dark-Flare":   0.28,
    "Medium-Flare": 0.28,
    "Light-Flare":  0.28,
    "Dark-Smoke":   0.42,
    "Medium-Smoke": 0.42,
    "Light-Smoke":  0.45,
}


# ── Couleurs par qualité de combustion (BGR pour OpenCV) ─────────────────

QUALITY_COLORS = {
    "bonne":    (0, 200, 80),    # Vert
    "moyenne":  (0, 165, 255),   # Orange BGR  — MODIF-11 : corrigé (était (0,180,240) = jaune)
    "mauvaise": (0, 60, 230),    # Rouge BGR
    "inactive": (160, 160, 160), # Gris        — MODIF-1 : torchère non validée
    "ambient":  (200, 180, 80),  # Bleu ciel   — MODIF-3 : fumée ambiante non attribuée
}

QUALITY_EMOJIS = {
    "bonne":    "[OK]",
    "moyenne":  "[~~]",
    "mauvaise": "[!!]",
    "inactive": "[--]",  # MODIF-1
    "ambient":  "[~?]",  # MODIF-3
}


# ── Validation temporelle par torchère ────────────────────────────────────
# MODIF-1 : nouvelle classe FlareState

class FlareState:
    """
    Gère l'état actif/inactif d'une torchère avec hystérésis double.

    Règle 1 — Flamme obligatoire (MODIF-2) :
        update(flame_detected=False) ne peut jamais activer une torchère.
        La fumée seule est insuffisante pour déclarer une torchère active.

    Règle 2 — Hystérésis temporelle (MODIF-1) :
        Activation   : ACTIVATION_FRAMES  consécutives AVEC flamme → active
        Désactivation: DEACTIVATION_FRAMES consécutives SANS flamme → inactive

    L'asymétrie (DEACTIVATION_FRAMES > ACTIVATION_FRAMES) protège contre :
        - Occlusions brèves (oiseau, passage d'objet, poussière)
        - Scintillement de flamme lors d'allumage / extinction
        - Artefacts de compression vidéo (H.264/H.265 blocage)
        - Reflets solaires sur structures métalliques (1-3 frames max)
    """

    def __init__(self):
        self.active           = False
        self.flame_counter    = 0   # frames consécutives avec flamme
        self.no_flame_counter = 0   # frames consécutives sans flamme

    def update(self, flame_detected: bool) -> bool:
        """
        Met à jour l'état selon la présence de flamme dans le frame courant.

        Paramètres
        ----------
        flame_detected : bool
            True si au moins une détection *-Flare est présente dans le groupe.

        Retourne
        --------
        bool : True si la torchère est considérée active après cette mise à jour.
        """
        if flame_detected:
            self.flame_counter    += 1
            self.no_flame_counter  = 0
            if self.flame_counter >= ACTIVATION_FRAMES:
                self.active = True
        else:
            self.no_flame_counter += 1
            self.flame_counter     = 0
            if self.no_flame_counter >= DEACTIVATION_FRAMES:
                self.active = False
        return self.active


# ── Chargement optionnel du SVM entraîné ─────────────────────────────────
# MODIF-7 : nouvelle fonction

def charger_classificateur_svm(forcer_hsv=False):
    """
    Tente de charger le modèle SVM 113-features et son scaler.

    Retourne (svm, scaler, fn_features) si disponibles,
    sinon (None, None, None) → le pipeline bascule sur l'heuristique HSV.

    Paramètres
    ----------
    forcer_hsv : bool
        Si True, ignore le SVM et utilise toujours l'heuristique HSV.
        Utile pour comparer les deux approches ou déboguer.
    """
    if forcer_hsv:
        print("  [INFO] Mode --no-svm actif : heuristique HSV forcée")
        return None, None, None

    try:
        import joblib
        from sklearn.pipeline import Pipeline as SkPipeline
        from src.classification.features import extraire_features

        # Chercher le modèle dans les emplacements connus
        svm_candidates = [
            ROOT / "outputs" / "classification" / "svm" / "svm_model.pkl",
            ROOT / "outputs" / "classification" / "svm" / "best_svm.pkl",
            ROOT / "outputs" / "classification" / "svm" / "svm.pkl",
        ]

        svm_path = next((p for p in svm_candidates if p.exists()), None)

        if svm_path is None:
            print("  [WARN] Modèle SVM introuvable → fallback HSV")
            return None, None, None

        svm = joblib.load(str(svm_path))

        # MODIF-23 : le modèle sauvegardé peut être un Pipeline(scaler+SVC)
        # ou un SVC nu. On gère les deux cas.
        if isinstance(svm, SkPipeline):
            # Pipeline : le scaler est intégré, pas besoin de fichier séparé
            classes = list(svm.named_steps["svm"].classes_)
            print(f"  [OK]   SVM chargé  : {svm_path.name} (Pipeline)")
            print(f"         Steps       : {list(svm.named_steps.keys())}")
            print(f"         Classes     : {classes}")
            # scaler=None → analyser_qualite_combustion sait que Pipeline gère tout
            return svm, None, extraire_features
        else:
            # SVC nu : chercher un scaler séparé
            scaler_candidates = [
                ROOT / "outputs" / "classification" / "svm" / "scaler.pkl",
                ROOT / "outputs" / "classification" / "svm" / "scaler_svm.pkl",
            ]
            scaler_path = next((p for p in scaler_candidates if p.exists()), None)
            if scaler_path is None:
                print("  [WARN] Scaler SVM introuvable → fallback HSV")
                return None, None, None
            scaler = joblib.load(str(scaler_path))
            print(f"  [OK]   SVM chargé  : {svm_path.name}")
            print(f"         Classes     : {list(svm.classes_)}")
            return svm, scaler, extraire_features

    except ImportError as e:
        print(f"  [WARN] Import impossible ({e}) → fallback HSV")
        return None, None, None
    except Exception as e:
        print(f"  [WARN] Erreur SVM ({e}) → fallback HSV")
        return None, None, None


# ── Analyse qualité de combustion ────────────────────────────────────────

def analyser_qualite_combustion(roi_bgr, cls_name="",
                                 svm=None, scaler=None, fn_features=None):
    """
    Classifie la qualité de combustion d'une ROI.

    Ordre de priorité :
        1. SVM 113-features si disponible (F1-macro = 95.80%)  — MODIF-7
        2. Heuristique HSV à deux niveaux (fallback si SVM absent)

    Heuristique HSV — Niveau 1 (nom de classe YOLO) :
        Dark-*   → mauvaise
        Light-*  → bonne (vérifié par HSV)
        Medium-* → décision entièrement HSV
    Heuristique HSV — Niveau 2 (analyse couleur) :
        dark_ratio, flame_ratio, smoke_ratio, mean_intensity

    Paramètres
    ----------
    roi_bgr      : np.ndarray — Image BGR de la ROI
    cls_name     : str        — Classe YOLO ('Dark-Smoke', 'Light-Flare', ...)
    svm          : objet sklearn SVM entraîné (optionnel)
    scaler       : StandardScaler (optionnel)
    fn_features  : callable   — Fonction d'extraction des 113 features (optionnel)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "moyenne", {"erreur": "ROI vide", "source": "n/a"}

    # ── Chemin 1 : SVM 113-features (priorité absolue) ───────────────
    # MODIF-7
    if svm is not None and fn_features is not None:
        try:
            features = fn_features(roi_bgr)
            # MODIF-23 : si svm est un Pipeline (scaler intégré),
            # on appelle predict directement ; sinon on scale manuellement.
            if scaler is not None:
                features_scaled = scaler.transform([features])
                pred = svm.predict(features_scaled)[0]
            else:
                pred = svm.predict([features])[0]
            # MODIF-23 : le SVM peut retourner un int (0/1/2) ou un str.
            # Mapping identique à train_svm.py : QUALITY_LABELS = ["bonne", "moyenne", "mauvaise"]
            _SVM_INT_TO_LABEL = {0: "bonne", 1: "moyenne", 2: "mauvaise"}
            if isinstance(pred, (int, np.integer)):
                qualite_svm = _SVM_INT_TO_LABEL.get(int(pred), "moyenne")
            else:
                qualite_svm = str(pred).lower().strip()
            if qualite_svm not in ("bonne", "moyenne", "mauvaise"):
                qualite_svm = "moyenne"  # valeur de sécurité si label inattendu
            return qualite_svm, {"source": "svm"}
        except Exception:
            pass  # échec SVM → fallback HSV

    # ── Chemin 2 : heuristique HSV ───────────────────────────────────

    # Niveau 1 : signal YOLO (prioritaire)
    cls_lower = cls_name.lower()
    if "dark" in cls_lower:
        qualite_yolo = "mauvaise"
    elif "light" in cls_lower:
        qualite_yolo = "bonne"
    else:
        qualite_yolo = None  # Medium-* ou inconnu → laisser HSV décider

    # Niveau 2 : analyse couleur HSV
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total_pixels = max(h.size, 1)

    dark_ratio  = float(np.sum(v < 50)) / total_pixels
    flame_mask  = (h >= 5) & (h <= 35) & (s > 80) & (v > 100)
    flame_ratio = float(np.sum(flame_mask)) / total_pixels
    smoke_mask  = (s < 60) & (v > 30) & (v < 200)
    smoke_ratio = float(np.sum(smoke_mask)) / total_pixels
    mean_v      = float(np.mean(v)) / 255.0

    scores = {
        "dark_ratio":     round(dark_ratio, 3),
        "flame_ratio":    round(flame_ratio, 3),
        "smoke_ratio":    round(smoke_ratio, 3),
        "mean_intensity": round(mean_v, 3),
        "source":         "hsv",  # MODIF-7 : traçabilité de la source
    }

    if qualite_yolo == "mauvaise":
        return "mauvaise", scores

    elif qualite_yolo == "bonne":
        # MODIF-12 : seuil extrême manquant dans l'original
        if dark_ratio > 0.70:
            return "mauvaise", scores  # Light-* mais ROI étonnamment très sombre
        if dark_ratio > 0.40:
            return "moyenne", scores
        return "bonne", scores

    # Medium-* : décision entièrement HSV
    if dark_ratio > 0.30:
        return "mauvaise", scores
    if dark_ratio > 0.15 and smoke_ratio > 0.30:
        return "mauvaise", scores
    if flame_ratio > 0.15 and mean_v > 0.45:
        return "bonne", scores
    if mean_v > 0.5 and dark_ratio < 0.10:
        return "bonne", scores
    return "moyenne", scores


# ── Groupement spatial en torchères distinctes ───────────────────────────

def grouper_torcheres(detections, marge=0.08,
                      dist_max_smoke=DIST_MAX_SMOKE):
    """
    Groupe les détections en torchères distinctes.

    Stratégie :
        1. Grouper les *-Flare entre eux par chevauchement (Union-Find).
        2. Assigner chaque *-Smoke au groupe Flare le plus proche,
           UNIQUEMENT si la distance est ≤ dist_max_smoke.     — MODIF-3
           Au-delà du seuil → fumée ambiante (liste séparée).
        3. Si aucune *-Flare dans le frame :                   — MODIF-4
           chaque détection = groupe individuel (pas de fusion).
           FlareState.update(flame_detected=False) empêchera l'activation.

    Retourne
    --------
    tuple[list[list[int]], list[int]]
        (groupes, indices_fumee_ambiante)
        groupes            : liste de groupes, chacun = liste d'indices
        indices_fumee_ambiante : indices de détections non attribuées
    """
    n = len(detections)
    if n == 0:
        return [], []
    if n == 1:
        cls = detections[0]["cls_name"].lower()
        # MODIF-4 : une fumée seule sans flamme → groupe individuel, pas fusion
        return [[0]], []

    flare_idx = [i for i in range(n)
                 if "flare" in detections[i]["cls_name"].lower()]
    smoke_idx = [i for i in range(n)
                 if "smoke" in detections[i]["cls_name"].lower()]

    # MODIF-4 : fallback sans flamme → groupes séparés (était [list(range(n))])
    if not flare_idx:
        return [[i] for i in smoke_idx], []

    # ── Union-Find sur les Flares ─────────────────────────────────────
    parent = {i: i for i in flare_idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def expand(det):
        bw = det["x2"] - det["x1"]
        bh = det["y2"] - det["y1"]
        mx, my = bw * marge, bh * marge
        return (det["x1"] - mx, det["y1"] - my,
                det["x2"] + mx, det["y2"] + my)

    for i in flare_idx:
        ex1i, ey1i, ex2i, ey2i = expand(detections[i])
        for j in flare_idx:
            if j <= i:
                continue
            ex1j, ey1j, ex2j, ey2j = expand(detections[j])
            if (ex1i < ex2j and ex1j < ex2i and
                    ey1i < ey2j and ey1j < ey2i):
                # MODIF-22 : Empêcher le groupement abusif de torchères séparées superposées verticalement.
                # Si elles se chevauchent verticalement dans leurs coordonnées d'origine,
                # on exige qu'elles aient un chevauchement vertical significatif (au moins 15% de la hauteur
                # de la plus petite des deux flammes). S'il y a un écart vertical complet (overlap_y <= 0),
                # on refuse également le groupement.
                y1_max = max(detections[i]["y1"], detections[j]["y1"])
                y2_min = min(detections[i]["y2"], detections[j]["y2"])
                overlap_y = y2_min - y1_max
                
                h_i = detections[i]["y2"] - detections[i]["y1"]
                h_j = detections[j]["y2"] - detections[j]["y1"]
                min_h = min(h_i, h_j)
                
                should_union = True
                if overlap_y <= 0:
                    should_union = False
                else:
                    ratio_y = overlap_y / min_h
                    if ratio_y < 0.15:
                        should_union = False
                        
                if should_union:
                    union(i, j)

    groups = {}
    for i in flare_idx:
        root = find(i)
        groups.setdefault(root, []).append(i)

    group_list = sorted(groups.values(),
                        key=lambda g: min(detections[i]["x1"] for i in g))

    # ── Assigner Smoke avec seuil de distance ─────────────────────────
    # MODIF-3 : ajout du seuil dist_max_smoke (était sans limite)
    def centroide(idx):
        d = detections[idx]
        return (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2

    ambient_smoke_indices = []  # MODIF-3

    for s_idx in smoke_idx:
        scx, scy   = centroide(s_idx)
        best_dist  = float("inf")
        best_group = 0
        for g_idx, group in enumerate(group_list):
            for f_idx in group:
                if "flare" not in detections[f_idx]["cls_name"].lower():
                    continue
                fcx, fcy = centroide(f_idx)
                d = ((scx - fcx) ** 2 + (scy - fcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist  = d
                    best_group = g_idx

        # MODIF-3 : seuil de distance — au-delà → fumée ambiante
        if best_dist <= dist_max_smoke:
            group_list[best_group].append(s_idx)
        else:
            ambient_smoke_indices.append(s_idx)

    return group_list, ambient_smoke_indices  # MODIF-3 : retourne un tuple


# ── Qualité de combustion d'un groupe ────────────────────────────────────

def determiner_qualite_torchere(detections, indices):
    """
    Détermine la qualité d'un groupe (= 1 torchère).

    MODIF-8  : utilise det["qualite"] (SVM ou HSV).
    MODIF-21 : logique signal mixte (bonne + mauvaise → moyenne).
    MODIF-24 : correction par ratio de classes YOLO Dark-*.

    Le SVM classifie chaque crop ROI individuellement : une flamme brillante
    = "bonne", de la fumée noire = "mauvaise". Cependant, il peut mal classer
    des crops de fumée noire en vidéo (différence avec les données d'entraînement).

    MODIF-24 ajoute un garde-fou : si le ratio de détections Dark-* (Dark-Flare +
    Dark-Smoke) dans le groupe dépasse un seuil, on force la qualité vers "mauvaise"
    indépendamment des prédictions SVM individuelles.

    Seuils MODIF-24 :
        dark_ratio ≥ 0.40 → mauvaise (forte fumée noire visible par YOLO)
        dark_ratio ≥ 0.20 → au moins moyenne (fumée noire significative)
    """
    qualites  = [detections[i]["qualite"] for i in indices]  # MODIF-8
    cls_names = [detections[i]["cls_name"] for i in indices]  # MODIF-24

    # ── MODIF-24 : ratio de classes YOLO Dark-* ─────────────────────────
    n_total = len(cls_names)
    n_dark  = sum(1 for c in cls_names if "dark" in c.lower())
    dark_ratio = n_dark / max(n_total, 1)

    if dark_ratio >= 0.40:
        return "mauvaise"   # fumée noire massive → combustion clairement mauvaise

    # ── MODIF-21 : signal mixte SVM ────────────────────────────────────
    has_bonne    = "bonne"    in qualites
    has_mauvaise = "mauvaise" in qualites
    has_moyenne  = "moyenne"  in qualites

    if has_moyenne:
        return "moyenne"
    if has_bonne and has_mauvaise:
        return "moyenne"
    if has_mauvaise:
        return "mauvaise"

    # ── MODIF-24 : dark_ratio modéré → au moins moyenne ───────────────
    if dark_ratio >= 0.20:
        return "moyenne"

    return "bonne"


# ── Suivi temporel des torchères ─────────────────────────────────────────

class TrackerTorcheres:
    """
    Suivi par centroïde ancré pour torchères industrielles (structures fixes).

    Les positions sont mises à jour lentement (EMA α=0.05) car les torchères
    ne se déplacent pas physiquement.

    Améliorations v2 :
        MODIF-1  FlareState dédié par torchère (règles 1 & 2)
        MODIF-5  Oubli automatique après TRACKER_MAX_ABSENT frames absentes
        MODIF-6  Labels robustes au-delà de 26 torchères (AA, AB, BA, ...)

    Améliorations v3 :
        MODIF-16 Archive de re-identification : les torchères "perdues"
                 sont gardées en archive pendant ~10s. Si une nouvelle
                 détection apparaît à la même position, elle récupère
                 son ancien label au lieu d'en créer un nouveau.
        MODIF-17 État FlareState préservé à la ré-ID (active reste active).
    """

    def __init__(self, dist_max=300, alpha=0.05):
        self.connus      = []   # list[dict] — voir _new_entry()
        self.archive     = []   # MODIF-16 : pool de re-identification
        self.prochain_id = 0
        self.dist_max    = dist_max
        self.alpha       = alpha

    # ── Helpers privés ───────────────────────────────────────────────

    def _prochain_label(self):
        """
        Génère un label alphabétique unique, robuste au-delà de 26.

        MODIF-6 : l'original faisait chr(ord("A") + id) → caractère non-ASCII
        à partir de la 27e torchère (chr(91) = '[').

        A, B, ..., Z, AA, AB, ..., AZ, BA, ...
        """
        n = self.prochain_id
        if n < 26:
            return chr(ord("A") + n)
        return (chr(ord("A") + (n // 26) - 1) +
                chr(ord("A") + (n % 26)))

    def _new_entry(self, cx, cy, label):
        """Crée une nouvelle entrée tracker avec FlareState dédié."""
        return {
            "label":  label,           # MODIF-24 : peut être None (candidat sans flamme)
            "cx":     cx,
            "cy":     cy,
            "vues":   1,
            "absent": 0,               # MODIF-5  : compteur d'absences
            "state":  FlareState(),    # MODIF-1  : état actif/inactif dédié
            "has_seen_flame": False,   # MODIF-24 : True dès la 1ère flamme observée
        }

    def _centroide(self, detections, indices):
        cx = sum((detections[i]["x1"] + detections[i]["x2"]) / 2
                 for i in indices) / len(indices)
        cy = sum((detections[i]["y1"] + detections[i]["y2"]) / 2
                 for i in indices) / len(indices)
        return cx, cy

    # ── API publique ─────────────────────────────────────────────────

    def get_state(self, label_court: str) -> FlareState:
        """
        Retourne le FlareState d'une torchère par son label court ('A', 'AA', ...).

        MODIF-1 : nouvelle méthode pour accéder au FlareState depuis run_pipeline.
        """
        for tk in self.connus:
            if tk["label"] == label_court:
                return tk["state"]
        return FlareState()  # fallback orphelin (ne devrait pas arriver)

    def assigner_labels(self, detections, groupes, has_flame_per_group=None):
        """
        Associe chaque groupe de la frame courante à une torchère connue
        (matching glouton par distance centroïde croissante), ou crée
        une nouvelle entrée si aucun match n'est trouvé.

        MODIF-5  : incrémente les absences pour les torchères non matchées
                   et supprime les fantômes après TRACKER_MAX_ABSENT frames.
        MODIF-24 : allocation paresseuse du label. Tant qu'aucune flamme n'a
                   été observée dans l'historique d'un groupe (has_flame
                   toujours False), le groupe est un "candidat" : suivi en
                   interne mais sans label (retour None). Évite l'attribution
                   de labels A/B/C aux nuages de vapeur que YOLO hallucine
                   en *-Smoke. Le label est alloué à la première frame où
                   `has_flame_per_group[g_idx]` est True.

        Parameters
        ----------
        has_flame_per_group : list[bool] | None
            Pour chaque groupe, True si au moins une détection *-Flare est
            présente. None ⇒ tous False (rétro-compatible).

        Returns
        -------
        list[str | None]
            "Torchere X" pour groupes labellisés, None pour candidats.
        """
        if not groupes:
            # MODIF-5 : même sans groupe, incrémenter les absences
            for tk in self.connus:
                tk["absent"] += 1
            self._nettoyer_fantomes()
            return []

        # MODIF-24 : valeur par défaut si l'appelant ne fournit pas l'info
        if has_flame_per_group is None:
            has_flame_per_group = [False] * len(groupes)

        centroides = [self._centroide(detections, g) for g in groupes]

        paires = []
        for g_idx, (cx, cy) in enumerate(centroides):
            for t_idx, tk in enumerate(self.connus):
                dx = cx - tk["cx"]
                dy = cy - tk["cy"]
                d  = (dx * dx + dy * dy) ** 0.5
                paires.append((d, g_idx, t_idx))
        paires.sort()

        labels       = [None] * len(groupes)
        groupes_pris = set()
        connus_pris  = set()

        for dist, g_idx, t_idx in paires:
            if g_idx in groupes_pris or t_idx in connus_pris:
                continue
            if dist <= self.dist_max:
                tk = self.connus[t_idx]
                # MODIF-24 : allocation paresseuse à la 1ère flamme observée
                if has_flame_per_group[g_idx] and not tk["has_seen_flame"]:
                    tk["has_seen_flame"] = True
                    if tk["label"] is None:
                        tk["label"] = self._prochain_label()
                        self.prochain_id += 1
                labels[g_idx] = tk["label"]  # peut rester None si candidat
                tk["cx"]    += self.alpha * (centroides[g_idx][0] - tk["cx"])
                tk["cy"]    += self.alpha * (centroides[g_idx][1] - tk["cy"])
                tk["vues"]  += 1
                tk["absent"] = 0  # MODIF-5 : reset du compteur d'absences
                groupes_pris.add(g_idx)
                connus_pris.add(t_idx)

        # MODIF-5 : incrémenter les absences des torchères non matchées ce frame
        for t_idx in range(len(self.connus)):
            if t_idx not in connus_pris:
                self.connus[t_idx]["absent"] += 1

        # Groupes non matchés → tenter ré-identification dans l'archive,
        # sinon créer une nouvelle torchère (ou un candidat sans flamme).
        # MODIF-16 : ré-identification depuis l'archive
        # MODIF-20 : vérifier que le label archivé n'est pas déjà pris
        # MODIF-24 : seuls les groupes avec flamme peuvent réveiller l'archive
        #            ou consommer un nouveau label ; sinon → candidat (label=None)
        for g_idx in range(len(groupes)):
            if g_idx in groupes_pris:
                continue
            cx, cy = centroides[g_idx]

            # MODIF-20 : collecter les labels déjà en usage ce frame
            labels_en_usage = set(l for l in labels if l is not None)
            labels_en_usage.update(
                tk["label"] for tk in self.connus if tk["label"] is not None
            )

            # Chercher dans l'archive (seulement si une flamme est présente)
            best_archive      = None
            best_archive_dist = float("inf")
            if has_flame_per_group[g_idx]:
                reid_radius = self.dist_max * TRACKER_REID_DIST_FACTOR
                for a_entry in self.archive:
                    # MODIF-20 : ignorer les entrées dont le label est déjà pris
                    if a_entry["label"] in labels_en_usage:
                        continue
                    dx = cx - a_entry["cx"]
                    dy = cy - a_entry["cy"]
                    d  = (dx * dx + dy * dy) ** 0.5
                    if d <= reid_radius and d < best_archive_dist:
                        best_archive_dist = d
                        best_archive      = a_entry

            if best_archive is not None:
                # Re-identification : récupérer le label d'origine
                lettre = best_archive["label"]
                self.archive.remove(best_archive)
                entry = self._new_entry(cx, cy, lettre)
                entry["has_seen_flame"] = True   # MODIF-24
                entry["vues"] = best_archive["vues"] + 1
                # MODIF-17 : préserver l'état actif si la torchère l'était
                if best_archive["state"].active:
                    entry["state"].active        = True
                    entry["state"].flame_counter = ACTIVATION_FRAMES
                labels[g_idx] = lettre
            elif has_flame_per_group[g_idx]:
                # Vraiment nouvelle torchère avec flamme
                lettre = self._prochain_label()  # MODIF-6
                self.prochain_id += 1
                entry = self._new_entry(cx, cy, lettre)
                entry["has_seen_flame"] = True   # MODIF-24
                labels[g_idx] = lettre
            else:
                # MODIF-24 : candidat sans flamme — suivi mais pas de label
                entry = self._new_entry(cx, cy, None)
                labels[g_idx] = None

            self.connus.append(entry)

        # MODIF-19 : filet de sécurité — détecter et corriger toute
        # duplication de label dans la liste retournée. En théorie impossible
        # (matching 1-vers-1 garanti, archives uniques, _prochain_label
        # monotone), mais on protège en profondeur contre les régressions.
        # MODIF-24 : on ignore les candidats (label=None).
        seen = set()
        for g_idx, lettre in enumerate(labels):
            if lettre is None:
                continue
            if lettre in seen:
                nouveau = self._prochain_label()
                self.prochain_id += 1
                labels[g_idx] = nouveau
                # Mettre à jour l'entrée connus correspondante
                target_cx, target_cy = centroides[g_idx]
                for tk in self.connus:
                    if (abs(tk["cx"] - target_cx) < 5
                            and abs(tk["cy"] - target_cy) < 5
                            and tk["label"] == lettre):
                        tk["label"] = nouveau
                        break
                lettre = nouveau
            seen.add(lettre)

        self._nettoyer_fantomes()  # MODIF-5
        # MODIF-24 : None pour les candidats sans flamme observée
        return [f"Torchere {l}" if l is not None else None for l in labels]

    def _nettoyer_fantomes(self):
        """
        Déplace les fantômes dans l'archive (re-ID) plutôt que de les supprimer.
        MODIF-5  : l'original ne supprimait jamais rien → mémoire croissante.
        MODIF-16 : v3 — archive au lieu de delete pour récupérer le label.

        Les entrées trop vieilles dans l'archive (>TRACKER_ARCHIVE_MAX_AGE)
        sont définitivement oubliées.
        """
        # Déplacer connus expirés vers l'archive
        vivants = []
        for tk in self.connus:
            if tk["absent"] <= TRACKER_MAX_ABSENT:
                vivants.append(tk)
            elif tk["label"] is None:
                # MODIF-24 : candidat sans flamme jamais labellisé → rien à
                # ré-identifier. On l'oublie au lieu de l'archiver, sinon une
                # vraie flamme apparaissant à proximité pourrait hériter de
                # son label=None et devenir invisible.
                pass
            else:
                # MODIF-16 : archiver l'entrée pour re-ID ultérieure
                # MODIF-20 : dédupliquer — supprimer l'ancienne entrée avec
                # le même label avant d'en ajouter une fraîche
                self.archive = [a for a in self.archive
                                if a["label"] != tk["label"]]
                self.archive.append({**tk, "archive_age": 0})
        self.connus = vivants

        # Vieillir l'archive et purger les entrées trop anciennes
        for a in self.archive:
            a["archive_age"] += 1
        self.archive = [a for a in self.archive
                        if a["archive_age"] <= TRACKER_ARCHIVE_MAX_AGE]


# ── Overlay visuel ───────────────────────────────────────────────────────

def dessiner_detection(frame, x1, y1, x2, y2, cls_name, conf, qualite,
                       torchere_label=""):
    """
    Dessine une bounding box annotée sur la frame.

    MODIF-1  : supporte qualite='inactive' (trait fin, texte réduit, pas de barre)
    MODIF-3  : supporte qualite='ambient'
    MODIF-12 : clip de la barre qualité pour éviter dépassement bas de frame
    """
    color = QUALITY_COLORS.get(qualite, (255, 255, 255))
    emoji = QUALITY_EMOJIS.get(qualite, "")

    # MODIF-1 : trait plus fin pour les états non actifs
    thickness = 1 if qualite in ("inactive", "ambient") else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    prefix     = f"{torchere_label} | " if torchere_label else ""
    label      = f"{prefix}{cls_name} {conf:.0%} | {qualite} {emoji}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 if qualite in ("inactive", "ambient") else 0.55  # MODIF-1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

    label_y1 = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # MODIF-1 / MODIF-12 : barre qualité uniquement si actif, avec clip vertical
    if qualite not in ("inactive", "ambient"):
        bar_h  = 4
        fh     = frame.shape[0]
        y2_clp = min(y2, fh - 1)               # MODIF-12 : clip
        y2_bar = min(y2_clp + bar_h, fh - 1)   # MODIF-12 : clip
        cv2.rectangle(frame, (x1, y2_clp), (x2, y2_bar), color, -1)

    return frame


def dessiner_hud(frame, fps, frame_idx, total_frames, stats_qualite,
                 stats_ambient=0, stats_inactive=0):
    """
    Dessine le HUD avec FPS, progression, stats qualité.

    MODIF-1/MODIF-3 : ajout des compteurs ambient et inactive.
    """
    h_frm, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (30, 30, 30), -1)  # MODIF-1 : hauteur +7
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "SONATRACH - Surveillance Torcheres", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    progress = f"Frame {frame_idx}"
    if total_frames > 0:
        pct = frame_idx / total_frames * 100
        progress += f"/{total_frames} ({pct:.0f}%)"
    cv2.putText(frame, f"FPS: {fps:.1f} | {progress}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # MODIF-1/MODIF-3 : ligne 3 — compteurs ambient et inactive
    cv2.putText(frame,
                f"ambient: {stats_ambient}  |  inactive: {stats_inactive}",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

    x_right = w - 200
    y_start = 20
    for i, (q, count) in enumerate(stats_qualite.items()):
        color = QUALITY_COLORS.get(q, (255, 255, 255))
        cv2.putText(frame, f"{q}: {count}", (x_right, y_start + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame


def dessiner_recap_final(w, h, stats_par_torchere, stats_qualite,
                         total_detections, frame_count, fps_moyen,
                         stats_ambient=0, stats_inactive=0):
    """
    Génère la frame de récapitulatif final.

    MODIF-1/MODIF-3 : intègre les compteurs ambient et inactive
    dans l'en-tête et les barres de qualité par torchère.
    """
    # MODIF-26 : n'afficher que les torchères retenues (≥ MIN_DISPLAY_ACTIVE détections
    # actives), comme le récap console — on masque les transitoires/fantômes.
    stats_par_torchere = {
        tname: tq for tname, tq in stats_par_torchere.items()
        if sum(v for k, v in tq.items() if k != "inactive") >= MIN_DISPLAY_ACTIVE
    }

    frame    = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    font     = cv2.FONT_HERSHEY_SIMPLEX
    y        = 40

    cv2.putText(frame, "SONATRACH - RECAPITULATIF", (20, y),
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    y += 15
    cv2.line(frame, (20, y), (w - 20, y), (100, 100, 100), 1)
    y += 35

    # MODIF-1/MODIF-3 : ajout ambient et inactive dans les stats globales
    cv2.putText(frame,
                f"Frames: {frame_count}  |  FPS: {fps_moyen:.1f}  |  "
                f"Detections: {total_detections}  |  "
                f"Torcheres: {len(stats_par_torchere)}  |  "
                f"Ambient: {stats_ambient}  |  Inactif: {stats_inactive}",
                (20, y), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    y += 40

    for tname in sorted(stats_par_torchere.keys()):
        tq       = stats_par_torchere[tname]
        # MODIF-1 : exclure "inactive" du calcul de la qualité dominante active
        tq_actif = {k: v for k, v in tq.items() if k != "inactive"}
        t_total  = sum(tq_actif.values())
        if t_total == 0:
            dominant = "inactive"
            pct_dom  = 100.0
        else:
            dominant = max(tq_actif, key=tq_actif.get)
            pct_dom  = tq_actif[dominant] / t_total * 100
        color = QUALITY_COLORS.get(dominant, (255, 255, 255))

        cv2.putText(frame, f"{tname}", (20, y),
                    font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"  ->  {dominant.upper()} ({pct_dom:.0f}%)",
                    (220, y), font, 0.65, color, 2, cv2.LINE_AA)
        y += 30

        bar_x     = 40
        bar_w_max = w - 80
        bar_h     = 16
        # MODIF-1 : inclure "inactive" dans les barres
        for q in ["bonne", "moyenne", "mauvaise", "inactive"]:
            c       = tq.get(q, 0)
            t_all   = sum(tq.values())
            p       = c / max(t_all, 1)
            q_color = QUALITY_COLORS.get(q, (200, 200, 200))
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w_max, y + bar_h),
                          (60, 60, 60), -1)
            fill_w = int(bar_w_max * p)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, y),
                              (bar_x + fill_w, y + bar_h), q_color, -1)
            cv2.putText(frame, f"{q}: {c} ({p * 100:.1f}%)",
                        (bar_x + 5, y + 13), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y += bar_h + 4
        y += 15

        if y > h - 40:
            break

    return frame


def dessiner_panel_torcheres(frame, torcheres_info):
    """
    Dessine un panneau en bas de frame avec l'état de chaque torchère.
    Inclut désormais les torchères inactives (MODIF-1).
    """
    if not torcheres_info:
        return frame

    h_frame, w_frame = frame.shape[:2]
    panel_h = 40
    y_start = h_frame - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w_frame, h_frame), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    x_offset = 10
    for info in torcheres_info:
        color = QUALITY_COLORS.get(info["qualite"], (255, 255, 255))
        emoji = QUALITY_EMOJIS.get(info["qualite"], "")
        text  = f"{info['label']}: {info['qualite']} {emoji}"
        cv2.putText(frame, text, (x_offset, y_start + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x_offset += tw + 30

    return frame


# ── Pipeline principal ───────────────────────────────────────────────────

def trouver_meilleur_modele():
    """Cherche automatiquement le meilleur modèle YOLO disponible dans le projet."""
    candidats = [
        ROOT / "outputs" / "models" / "gas_flare_yolo11m_v1" / "weights" / "best.pt",
        ROOT / "outputs" / "models" / "gas_flare_yolov8m_v3" / "weights" / "best.pt",
        ROOT / "outputs" / "models" / "gas_flare_yolov8m_v2" / "weights" / "best.pt",
        ROOT / "outputs" / "models" / "gas_flare_yolov8s"     / "weights" / "best.pt",
    ]
    for candidat in candidats:
        if candidat.exists():
            return str(candidat)

    fallback_locaux = [ROOT / "yolo11m.pt", ROOT / "yolov8m.pt", ROOT / "yolov8s.pt"]
    for fallback in fallback_locaux:
        if fallback.exists():
            return str(fallback)

    return "yolo11m.pt"


def run_pipeline(source, model_path=None, conf=0.4,
                 output_dir=None, show=False, no_svm=False):
    """
    Lance le pipeline de surveillance temps réel.

    Paramètres
    ----------
    source     : str   — Chemin vidéo ou '0' pour webcam
    model_path : str   — Chemin modèle YOLO (.pt). Auto-détecté si None.
    conf       : float — Seuil de confiance détection (défaut: 0.4)
    output_dir : str   — Répertoire de sortie (défaut: outputs/realtime/)
    show       : bool  — Afficher fenêtre OpenCV en temps réel
    no_svm     : bool  — Forcer heuristique HSV, ignorer le SVM — MODIF-7
    """
    from ultralytics import YOLO

    # ── Configuration ─────────────────────────────────────────────────
    if model_path is None:
        model_path = trouver_meilleur_modele()
    if output_dir is None:
        output_dir = ROOT / "outputs" / "realtime"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  SONATRACH — Pipeline Surveillance Torchères v3")
    print("=" * 65)
    print(f"  Modèle           : {model_path}")
    print(f"  Source           : {source}")
    print(f"  Conf fallback    : {conf}")
    print(f"  Output           : {output_dir}")
    print(f"  Activation       : {ACTIVATION_FRAMES} frames consécutives")
    print(f"  Désactivation    : {DEACTIVATION_FRAMES} frames consécutives")
    print(f"  Tracker max abs. : {TRACKER_MAX_ABSENT} frames avant archive")
    print(f"  Archive max age  : {TRACKER_ARCHIVE_MAX_AGE} frames avant oubli")
    print(f"  Dist max smoke   : {DIST_MAX_SMOKE} px")
    # MODIF-13 : afficher les seuils par classe
    print(f"  Seuils per-classe:")
    for cls, s in CLASS_CONF_MIN.items():
        print(f"     {cls:<14} : {s}")

    # ── MODIF-7 : charger le SVM avant YOLO ──────────────────────────
    print("\n[1/4] Chargement des modèles...")
    svm, scaler, fn_features = charger_classificateur_svm(forcer_hsv=no_svm)
    mode_classif = "SVM 113-features" if svm is not None else "Heuristique HSV"
    print(f"  Classification : {mode_classif}")

    model       = YOLO(model_path)
    class_names = model.names
    print(f"  YOLO classes   : {class_names}")

    # ── Ouvrir la vidéo ───────────────────────────────────────────────
    print("\n[2/4] Ouverture de la source vidéo...")
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir : {source}")
        return

    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Résolution : {w}×{h}")
    print(f"  FPS source : {fps_video:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Durée      : {total_frames / fps_video:.1f}s")

    # ── Préparer la sortie ────────────────────────────────────────────
    output_video_path = output_dir / f"realtime_output_{timestamp_str}.mp4"
    fourcc            = cv2.VideoWriter_fourcc(*"mp4v")
    writer            = cv2.VideoWriter(str(output_video_path), fourcc, fps_video, (w, h))

    csv_path   = output_dir / f"detections_log_{timestamp_str}.csv"
    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    # MODIF-7 : colonne classif_source (svm / hsv / n/a)
    # MODIF-1/MODIF-3 : qualite peut valoir inactive ou ambient_smoke
    csv_writer.writerow([
        "frame_idx", "timestamp_sec", "torchere", "class_name", "confidence",
        "qualite",        # bonne | moyenne | mauvaise | inactive | ambient_smoke
        "x1", "y1", "x2", "y2",
        "dark_ratio", "flame_ratio", "smoke_ratio", "mean_intensity",
        "classif_source", # MODIF-7 : svm | hsv | n/a
    ])

    # ── Boucle d'inférence ────────────────────────────────────────────
    print("\n[3/4] Traitement en cours...")

    stats_qualite      = {"bonne": 0, "moyenne": 0, "mauvaise": 0}
    stats_par_torchere = {}
    total_detections   = 0
    stats_ambient      = 0  # MODIF-3
    stats_inactive     = 0  # MODIF-1
    fps_history        = []
    frame_idx          = 0
    tracker            = TrackerTorcheres(dist_max=max(w, h) // 3)

    # MODIF-9 : writer.release() déplacé dans le finally externe
    try:
        # ── Boucle principale (cap + csv fermés dans finally interne) ─
        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx    += 1
                timestamp_sec = frame_idx / fps_video

                # Inférence YOLO avec seuil bas, filtrage précis post-process
                # MODIF-14 : conf=0.20 (global bas) + filtre per-classe ci-après
                results = model.predict(
                    frame, conf=0.20, iou=0.45,
                    verbose=False, stream=False,
                )[0]

                # ── Pass 1 : collecter + qualifier chaque détection ──────
                # MODIF-7  : passer svm/scaler/fn_features à analyser_qualite_combustion
                # MODIF-8  : det["qualite"] est maintenant SVM ou HSV, non plus ignoré
                # MODIF-13 : filtrage de confiance par classe (au lieu d'un seuil global)
                frame_detections = []
                for box in results.boxes:
                    cls_id     = int(box.cls[0])
                    cls_name   = class_names[cls_id]
                    confidence = float(box.conf[0])

                    # MODIF-13 : filtre per-classe (capture petites flammes,
                    # rejette fumées fantômes)
                    seuil_classe = CLASS_CONF_MIN.get(cls_name, conf)
                    if confidence < seuil_classe:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    roi = frame[y1:y2, x1:x2]

                    qualite_ind, scores = analyser_qualite_combustion(
                        roi, cls_name=cls_name,
                        svm=svm, scaler=scaler, fn_features=fn_features)  # MODIF-7

                    frame_detections.append({
                        "cls_name": cls_name, "conf": confidence,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "qualite": qualite_ind,  # MODIF-8 : utilisé par determiner_qualite_torchere
                        "scores":  scores,
                    })

                # ── Pass 2 : groupement + fumée ambiante ────────────────
                # MODIF-3/MODIF-4 : grouper_torcheres retourne maintenant un tuple
                groupes, ambient_indices = grouper_torcheres(frame_detections)

                # MODIF-26 : calculer has_flame_per_group pour allocation
                # paresseuse des labels (Niveau 2 — pas de label sans flamme)
                has_flame_per_group = [
                    any("flare" in frame_detections[i]["cls_name"].lower()
                        for i in g)
                    for g in groupes
                ]
                labels_stables = tracker.assigner_labels(
                    frame_detections, groupes, has_flame_per_group)
                torcheres_info           = []

                # ── Loguer la fumée ambiante ─────────────────────────────
                # MODIF-3 : fumées non attribuées à une torchère
                for amb_idx in ambient_indices:
                    det = frame_detections[amb_idx]
                    stats_ambient += 1
                    dessiner_detection(
                        frame, det["x1"], det["y1"], det["x2"], det["y2"],
                        det["cls_name"], det["conf"], "ambient",
                        torchere_label="")
                    csv_writer.writerow([
                        frame_idx, f"{timestamp_sec:.3f}",
                        "ambient",       # MODIF-3 : pas de torchère associée
                        det["cls_name"], f"{det['conf']:.4f}",
                        "ambient_smoke", # MODIF-3
                        det["x1"], det["y1"], det["x2"], det["y2"],
                        det["scores"].get("dark_ratio", ""),
                        det["scores"].get("flame_ratio", ""),
                        det["scores"].get("smoke_ratio", ""),
                        det["scores"].get("mean_intensity", ""),
                        det["scores"].get("source", "n/a"),  # MODIF-7
                    ])

                # ── Traiter chaque groupe de torchère ────────────────────
                for g_idx, indices in enumerate(groupes):
                    label_full = labels_stables[g_idx]   # "Torchere A" ou None

                    # MODIF-26 (Niveau 2) : candidat sans flamme → skip
                    # Le tracker suit le groupe en interne mais pas de label,
                    # pas de dessin, pas de log CSV.
                    if label_full is None:
                        continue

                    label_court = label_full.split()[-1]  # "A"

                    # MODIF-2 : Règle 1 — flamme obligatoire pour activation
                    has_flame = any(
                        "flare" in frame_detections[i]["cls_name"].lower()
                        for i in indices
                    )

                    # MODIF-1 : Règle 2 — validation temporelle (hystérésis)
                    state     = tracker.get_state(label_court)
                    is_active = state.update(has_flame)

                    # Initialiser stats par torchère
                    if label_full not in stats_par_torchere:
                        stats_par_torchere[label_full] = {
                            "bonne": 0, "moyenne": 0, "mauvaise": 0,
                            "inactive": 0,  # MODIF-1
                        }

                    if not is_active:
                        # ── Torchère inactive (règle 1 ou 2 non satisfaite) ──
                        # MODIF-1 : loguée comme "inactive", pas de classification qualité
                        torcheres_info.append({
                            "label":        label_full,
                            "qualite":      "inactive",
                            "n_detections": len(indices),
                        })
                        # MODIF-26 (Niveau 1) : plus de détection primaire à
                        # calculer ici — les inactives ne sont plus dessinées.
                        for det_idx in indices:
                            det = frame_detections[det_idx]
                            stats_inactive += 1
                            stats_par_torchere[label_full]["inactive"] += 1
                            # MODIF-26 (Niveau 1) : ne plus dessiner les
                            # détections inactives (pas de bbox grise ni label).
                            # Le CSV conserve la trace pour audit.
                            csv_writer.writerow([
                                frame_idx, f"{timestamp_sec:.3f}",
                                label_full, det["cls_name"],
                                f"{det['conf']:.4f}",
                                "inactive",  # MODIF-1
                                det["x1"], det["y1"], det["x2"], det["y2"],
                                det["scores"].get("dark_ratio", ""),
                                det["scores"].get("flame_ratio", ""),
                                det["scores"].get("smoke_ratio", ""),
                                det["scores"].get("mean_intensity", ""),
                                det["scores"].get("source", "n/a"),  # MODIF-7
                            ])
                        continue  # MODIF-1 : pas de classification qualité active

                    # ── Torchère active : classification qualité ──────────
                    # MODIF-8 : utilise det["qualite"] (SVM ou HSV) via determiner_qualite_torchere
                    qualite_groupe = determiner_qualite_torchere(frame_detections, indices)
                    torcheres_info.append({
                        "label":        label_full,
                        "qualite":      qualite_groupe,
                        "n_detections": len(indices),
                    })

                    # MODIF-18 : identifier la détection PRIMAIRE pour
                    # afficher le label "Torchere X" une seule fois par groupe.
                    # Priorité : flamme (l'ancre du groupe) > première détection.
                    primary_idx = next(
                        (i for i in indices
                         if "flare" in frame_detections[i]["cls_name"].lower()),
                        indices[0]
                    )

                    for det_idx in indices:
                        det = frame_detections[det_idx]
                        total_detections += 1
                        stats_qualite[qualite_groupe] += 1
                        stats_par_torchere[label_full][qualite_groupe] += 1

                        # MODIF-18 : label uniquement sur la détection primaire
                        label_drawn = label_full if det_idx == primary_idx else ""

                        dessiner_detection(
                            frame, det["x1"], det["y1"],
                            det["x2"], det["y2"],
                            det["cls_name"], det["conf"],
                            qualite_groupe, torchere_label=label_drawn)

                        csv_writer.writerow([
                            frame_idx, f"{timestamp_sec:.3f}",
                            label_full, det["cls_name"],
                            f"{det['conf']:.4f}",
                            qualite_groupe,
                            det["x1"], det["y1"], det["x2"], det["y2"],
                            det["scores"].get("dark_ratio", ""),
                            det["scores"].get("flame_ratio", ""),
                            det["scores"].get("smoke_ratio", ""),
                            det["scores"].get("mean_intensity", ""),
                            det["scores"].get("source", "n/a"),  # MODIF-7
                        ])

                # ── Panneau, HUD, sauvegarde ─────────────────────────────
                if len(torcheres_info) > 1:
                    dessiner_panel_torcheres(frame, torcheres_info)

                elapsed     = time.time() - t0
                fps_current = 1.0 / max(elapsed, 1e-6)
                fps_history.append(fps_current)

                # MODIF-1/MODIF-3 : passer les nouveaux compteurs au HUD
                dessiner_hud(frame, fps_current, frame_idx, total_frames,
                             stats_qualite, stats_ambient, stats_inactive)

                writer.write(frame)

                if show:
                    cv2.imshow("Sonatrach - Surveillance Torcheres", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("  [INFO] Arrêt par l'utilisateur (touche Q)")
                        break

                if frame_idx % 100 == 0:
                    avg_fps = sum(fps_history[-100:]) / min(len(fps_history), 100)
                    pct     = frame_idx / max(total_frames, 1) * 100
                    print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%) "
                          f"| FPS: {avg_fps:.1f} "
                          f"| Actives: {total_detections} "
                          f"| Ambient: {stats_ambient} "     # MODIF-3
                          f"| Inactif: {stats_inactive}")    # MODIF-1

        except KeyboardInterrupt:
            print("\n  [INFO] Arrêt par Ctrl+C")
        finally:
            cap.release()
            csv_file.close()
            if show:
                cv2.destroyAllWindows()

        # ── Écran récapitulatif (3 secondes ajoutées à la vidéo) ─────
        # Exécuté après la fermeture de cap/csv mais avant writer.release()
        fps_moyen = sum(fps_history) / max(len(fps_history), 1)  # MODIF-10 : calculé ici, une seule fois
        if stats_par_torchere:
            recap_frame     = dessiner_recap_final(
                w, h, stats_par_torchere, stats_qualite,
                total_detections, frame_idx, fps_moyen,
                stats_ambient, stats_inactive)               # MODIF-1/MODIF-3
            nb_recap_frames = int(fps_video * 3)
            for _ in range(nb_recap_frames):
                writer.write(recap_frame)

    finally:
        writer.release()  # MODIF-9 : toujours libéré, même en cas d'exception

    # ── Résultats console ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  [4/4] RÉSULTATS")
    print("=" * 65)
    print(f"  Frames traitées     : {frame_idx}")
    print(f"  FPS moyen           : {fps_moyen:.1f}")
    print(f"  Détections actives  : {total_detections}")
    print(f"  Détections inactives: {stats_inactive}")   # MODIF-1
    print(f"  Fumée ambiante      : {stats_ambient}")    # MODIF-3
    print(f"  Torchères détectées : {len(stats_par_torchere)}")
    print(f"  Classification      : {mode_classif}")     # MODIF-7

    print(f"\n  Distribution qualité (détections actives) :")
    for q in ["bonne", "moyenne", "mauvaise"]:
        count = stats_qualite[q]
        pct   = count / max(total_detections, 1) * 100
        bar   = "█" * int(pct / 2)
        print(f"    {q:<12}: {count:>5} ({pct:>5.1f}%)  {bar}")

    if stats_par_torchere:
        # MODIF-25 : filtrer les torchères fantômes (< MIN_DISPLAY_ACTIVE détections actives)
        torcheres_valides = {}
        torcheres_fantomes = []
        for tname, tq in stats_par_torchere.items():
            tq_actif = {k: v for k, v in tq.items() if k != "inactive"}
            t_actif = sum(tq_actif.values())
            if t_actif >= MIN_DISPLAY_ACTIVE:
                torcheres_valides[tname] = tq
            else:
                torcheres_fantomes.append((tname, sum(tq.values())))

        if torcheres_valides:
            print(f"\n  {'─' * 55}")
            print(f"  DÉTAIL PAR TORCHÈRE")
            print(f"  {'─' * 55}")
            emojis = {"bonne": "🟢", "moyenne": "🟡", "mauvaise": "🔴", "inactive": "⚪"}
            for tname in sorted(torcheres_valides.keys()):
                tq       = torcheres_valides[tname]
                # MODIF-1 : dominant calculé hors "inactive"
                tq_actif = {k: v for k, v in tq.items() if k != "inactive"}
                t_actif  = sum(tq_actif.values())
                if t_actif == 0:
                    dominant = "inactive"
                else:
                    dominant = max(tq_actif, key=tq_actif.get)
                pct_dom = tq.get(dominant, 0) / max(sum(tq.values()), 1) * 100
                print(f"\n  {emojis.get(dominant, '')} {tname}  "
                      f"→  {dominant.upper()} ({pct_dom:.0f}%)")
                print(f"    Détections totales : {sum(tq.values())}")
                for q in ["bonne", "moyenne", "mauvaise", "inactive"]:  # MODIF-1
                    c   = tq.get(q, 0)
                    p   = c / max(sum(tq.values()), 1) * 100
                    bar = "█" * int(p / 2)
                    print(f"      {q:<10}: {c:>4} ({p:>5.1f}%)  {bar}")
            if torcheres_fantomes:
                print(f"\n  ({len(torcheres_fantomes)} torchère(s) transitoire(s) "
                      f"filtrée(s) : < {MIN_DISPLAY_ACTIVE} détections actives)")
            print(f"  {'─' * 55}")

    print(f"\n  Vidéo sortie : {output_video_path}")
    print(f"  Log CSV      : {csv_path}")
    print("=" * 65)

    return {
        "frames":          frame_idx,
        "fps_moyen":       round(fps_moyen, 1),
        "detections":      total_detections,
        "ambient":         stats_ambient,   # MODIF-3
        "inactive":        stats_inactive,  # MODIF-1
        "qualite":         stats_qualite,
        "classif_source":  mode_classif,    # MODIF-7
        "video":           str(output_video_path),
        "csv":             str(csv_path),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline surveillance torchères v2 — Sonatrach PFE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python src/realtime_monitor.py --source data/test.mp4
  python src/realtime_monitor.py --source data/test.mp4 --show
  python src/realtime_monitor.py --source 0 --conf 0.5
  python src/realtime_monitor.py --source data/test.mp4 --no-svm
        """,
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Chemin vidéo ou '0' pour webcam",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Chemin modèle YOLO (.pt). Auto-détecté si non spécifié.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Seuil confiance FALLBACK (classes hors CLASS_CONF_MIN). "
             "Les *-Flare/*-Smoke utilisent leurs propres seuils — MODIF-13",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Répertoire de sortie (défaut: outputs/realtime/)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Afficher la fenêtre OpenCV en temps réel",
    )
    # MODIF-7 : nouveau flag --no-svm
    parser.add_argument(
        "--no-svm", action="store_true",
        help="Forcer l'heuristique HSV (ignorer le SVM entraîné)",
    )

    args = parser.parse_args()
    run_pipeline(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        output_dir=args.output,
        show=args.show,
        no_svm=args.no_svm,  # MODIF-7
    )


if __name__ == "__main__":
    main()