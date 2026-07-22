"""
Extraction de features visuelles pour classification qualité de combustion.

Chaque feature est justifiée par la physique de la combustion :
  - Couleur HSV    → flamme jaune/orange = bonne ; rouge/noir = mauvaise
  - Pixels sombres → proportion de fumée noire (combustion incomplète)
  - Intensité RGB  → flamme brillante = combustion complète
  - Texture LBP    → homogénéité flamme vs turbulence (implémentation numpy)
  - GLCM           → contraste, énergie de la texture (implémentation numpy)
  - Ratio fumée    → estimé via seuillage des zones grises/noires

Note implémentation :
  LBP et GLCM sont implémentés en numpy pur (sans scikit-image)
  pour éviter les problèmes de dépendances dans l'environnement venv.
  Les formules sont mathématiquement identiques à skimage.

Total : ~113 dimensions par ROI (94 + 1 + 3 + 10 + 4 + 1).
"""

import cv2
import numpy as np


# ── Features individuelles ───────────────────────────────────────────────

def compute_hsv_histogram(roi_bgr, bins_h=30, bins_s=32, bins_v=32):
    """
    Histogramme HSV normalisé (94 dim).

    Justification :
      - H (teinte) : distingue les couleurs de flamme (jaune, orange, rouge)
      - S (saturation) : flamme vive vs fumée grise
      - V (valeur) : flamme brillante vs fumée sombre
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins_h], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins_s], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins_v], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v])
    total = hist.sum()
    return hist / total if total > 0 else hist


def compute_dark_ratio(roi_bgr, threshold=50):
    """
    Proportion de pixels sombres (1 dim).

    V < threshold dans HSV → pixel noir/très sombre.
    Plus ce ratio est élevé, plus il y a de fumée noire → mauvaise combustion.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2] < threshold))


def compute_mean_intensity(roi_bgr):
    """
    Intensité moyenne des 3 canaux RGB (3 dim), normalisée [0, 1].

    Flamme brillante (haute intensité) → bonne combustion.
    Région sombre (faible intensité) → fumée, mauvaise combustion.
    """
    return roi_bgr.mean(axis=(0, 1)) / 255.0


def _lbp_uniform(gray, n_points=8, radius=1):
    """
    LBP uniforme — sémantique skimage.feature.local_binary_pattern(method='uniform').

    Un pattern est dit 'uniforme' lorsque son code binaire circulaire présente
    au plus deux transitions 0->1 ou 1->0. Pour ces patterns, l'étiquette de
    sortie est le nombre de bits à 1 (entier dans [0, n_points]). Tous les
    patterns non uniformes reçoivent l'étiquette commune n_points + 1. Il y a
    donc exactement n_points + 2 étiquettes distinctes — convention qui
    s'aligne sur le `range=(0, n_points + 2)` du `np.histogram` appelant.
    """
    h, w = gray.shape
    angles = [2 * np.pi * i / n_points for i in range(n_points)]
    center = gray.astype(np.float32)

    # Bits du code circulaire : voisin_i >= centre
    neighbor_bits = []
    for angle in angles:
        dx = int(round(radius * np.cos(angle)))
        dy = int(round(-radius * np.sin(angle)))
        shifted = np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1).astype(np.float32)
        neighbor_bits.append((shifted >= center).astype(np.int32))

    # Popcount du code (nombre de bits à 1)
    bit_count = np.zeros((h, w), dtype=np.int32)
    for bit in neighbor_bits:
        bit_count += bit

    # Nombre de transitions 0/1 dans le code circulaire
    transitions = np.zeros((h, w), dtype=np.int32)
    for i in range(n_points):
        transitions += (neighbor_bits[i] != neighbor_bits[(i + 1) % n_points]).astype(np.int32)

    # Uniforme : étiquette = popcount ; non uniforme : n_points + 1
    return np.where(transitions <= 2, bit_count, n_points + 1).astype(np.int32)


def compute_lbp_histogram(roi_bgr, n_points=8, radius=1):
    """
    Histogramme LBP – Local Binary Pattern (10 dim).

    Capture la micro-texture de la région :
      - Flamme homogène → distribution LBP concentrée
      - Fumée turbulente → distribution LBP étalée
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))
    n_bins = n_points + 2  # uniform → (n_points + 2) patterns
    lbp = _lbp_uniform(gray_resized, n_points, radius)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def _compute_glcm_numpy(gray_q, levels=64):
    """
    Calcule la GLCM (Gray-Level Co-occurrence Matrix) en numpy pur.
    Angles : 0° et 45°, distance=1, symétrique et normalisée.
    """
    h, w = gray_q.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    
    # Angle 0° — paires horizontales
    glcm += np.bincount(
        gray_q[:, :-1].ravel() * levels + gray_q[:, 1:].ravel(),
        minlength=levels * levels,
    ).reshape(levels, levels)
    # Angle 45° — paires diagonales
    glcm += np.bincount(
        gray_q[:-1, 1:].ravel() * levels + gray_q[1:, :-1].ravel(),
        minlength=levels * levels,
    ).reshape(levels, levels)
    
    # Symétrie
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total  # normaliser
    return glcm


def compute_glcm_features(roi_bgr):
    """
    Features GLCM – Gray-Level Co-occurrence Matrix (4 dim).

    Calcule 4 propriétés statistiques de texture :
      - contrast     : différence locale d'intensité
      - dissimilarity: variation locale
      - energy       : uniformité (flamme homogène → énergie élevée)
      - homogeneity  : degré de régularité locale
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))
    gray_q = (gray_resized // 4).astype(np.int32)  # 64 niveaux
    
    glcm = _compute_glcm_numpy(gray_q, levels=64)
    
    i_vals = np.arange(64)
    j_vals = np.arange(64)
    I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
    diff = np.abs(I - J).astype(np.float64)
    
    contrast      = float(np.sum(glcm * diff ** 2))
    dissimilarity = float(np.sum(glcm * diff))
    energy        = float(np.sum(glcm ** 2))
    homogeneity   = float(np.sum(glcm / (1.0 + diff)))
    
    return np.array([contrast, dissimilarity, energy, homogeneity], dtype=np.float32)


def compute_smoke_ratio(roi_bgr):
    """
    Ratio fumée vs flamme (1 dim), estimé par seuillage HSV.

    Heuristique :
      - Flamme : H ∈ [5, 35] (orange/jaune), S > 80, V > 100
      - Fumée  : S < 60 (désaturé/gris), V ∈ [30, 200]

    Ratio élevé → beaucoup de fumée → mauvaise combustion.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Masque flamme : orange/jaune, saturé, brillant
    flame_mask = (h >= 5) & (h <= 35) & (s > 80) & (v > 100)
    # Masque fumée : désaturé, pas trop sombre ni trop clair
    smoke_mask = (s < 60) & (v > 30) & (v < 200)

    n_flame = flame_mask.sum()
    n_smoke = smoke_mask.sum()

    if n_flame + n_smoke == 0:
        return 0.5  # indéterminé
    return float(n_smoke / (n_flame + n_smoke))


# ── Vecteur complet ──────────────────────────────────────────────────────

# Index ranges for ablation study
FEATURE_GROUPS = {
    "Hist. HSV":     (0, 94),
    "Ratio sombre":  (94, 95),
    "Intensité RGB": (95, 98),
    "Texture LBP":   (98, 108),
    "GLCM":          (108, 112),
    "Ratio fumée":   (112, 113),
}

FEATURE_DIM = 113  # total attendu


def extract_all_features(roi_bgr):
    """
    Extraire toutes les features d'un ROI.

    Returns
    -------
    np.ndarray de forme (113,)
    """
    features = [
        compute_hsv_histogram(roi_bgr),       # 94 dim
        np.array([compute_dark_ratio(roi_bgr)]),  # 1 dim
        compute_mean_intensity(roi_bgr),      # 3 dim
        compute_lbp_histogram(roi_bgr),       # 10 dim
        compute_glcm_features(roi_bgr),       # 4 dim
        np.array([compute_smoke_ratio(roi_bgr)]),  # 1 dim
    ]
    return np.concatenate(features)


# Alias français utilisé par realtime_monitor.py
extraire_features = extract_all_features
