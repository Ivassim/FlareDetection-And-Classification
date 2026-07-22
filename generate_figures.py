"""
Generation automatique des figures pour le memoire PFE
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_figs_dir() -> Path:
    """Résout automatiquement le dossier Figs du mémoire (ordre de priorité)."""
    candidates = [
        PROJECT_ROOT.parent / "Memoire" / "USTHB_Thesis - Copie (2)" / "Figs",
        PROJECT_ROOT.parent / "Memoire" / "USTHB_Thesis - Copie" / "Figs",
        PROJECT_ROOT.parent / "Memoire" / "USTHB_Thesis" / "Figs",
    ]
    for candidate in candidates:
        if candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate

    # Fallback : premier candidat si aucun dossier mémoire n'existe encore.
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


FIGS_DIR = resolve_figs_dir()
FIGS = str(FIGS_DIR)

DETECTION_RUN_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "models" / "gas_flare_yolo11m_v1",
    PROJECT_ROOT / "outputs" / "models" / "gas_flare_yolov8m_v3",
    PROJECT_ROOT / "outputs" / "models" / "gas_flare_yolov8m_v2",
    PROJECT_ROOT / "outputs" / "models" / "gas_flare_yolov8s",
]

SONATRACH_BLUE  = "#005BAA"
SONATRACH_GREEN = "#00A651"
ORANGE  = "#E87722"
RED     = "#C0392B"
YELLOW  = "#F1C40F"
GRAY    = "#7F8C8D"
DARK    = "#2C3E50"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# FIG 1 : MAPPING 6 CLASSES → 3 NIVEAUX
# ─────────────────────────────────────────────
def fig_mapping_classes():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    ax.text(5, 5.5, "Mapping des 6 classes YOLO vers 3 niveaux de qualité de combustion",
            ha='center', va='center', fontsize=13, fontweight='bold', color=DARK)

    yolo_classes = [
        ("Light-Flare",  "#F0F7FF", SONATRACH_BLUE),
        ("Light-Smoke",  "#E8F5E9", SONATRACH_GREEN),
        ("Medium-Flare", "#FFF3E0", ORANGE),
        ("Medium-Smoke", "#FFF8E1", YELLOW),
        ("Dark-Flare",   "#FFEBEE", RED),
        ("Dark-Smoke",   "#FCE4EC", "#880E4F"),
    ]
    quality = [
        ("BONNE\nCombustion", SONATRACH_GREEN, "Flamme claire\nCO₂ pur"),
        ("MOYENNE\nCombustion", ORANGE, "Flamme orange\nCombustion partielle"),
        ("MAUVAISE\nCombustion", RED, "Fumée noire\nCombustion incomplète"),
    ]

    for i, (name, bg, fg) in enumerate(yolo_classes):
        y = 4.2 - i * 0.7
        bbox = FancyBboxPatch((0.3, y-0.25), 2.8, 0.5,
                               boxstyle="round,pad=0.05", fc=bg, ec=fg, lw=1.5)
        ax.add_patch(bbox)
        ax.text(1.7, y, name, ha='center', va='center', color=DARK, fontsize=10, fontweight='bold')

    for i, (name, color, sub) in enumerate(quality):
        y = 3.6 - i * 1.4
        bbox = FancyBboxPatch((7.0, y-0.5), 2.7, 1.0,
                               boxstyle="round,pad=0.08", fc=color, ec='white', lw=2, alpha=0.9)
        ax.add_patch(bbox)
        ax.text(8.35, y+0.15, name, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        ax.text(8.35, y-0.25, sub, ha='center', va='center', color='white', fontsize=8)

    arrows = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]
    q_y = [3.6, 2.2, 0.8]
    src_y = [4.2 - i * 0.7 for i in range(6)]
    for src_i, dst_i in arrows:
        ax.annotate('', xy=(7.0, q_y[dst_i]), xytext=(3.1, src_y[src_i]),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5,
                                   connectionstyle='arc3,rad=0.1'))

    ax.text(5.0, 5.0, "Classes YOLO (6)", ha='center', color=SONATRACH_BLUE, fontsize=10, style='italic')
    ax.text(8.35, 5.0, "Qualité (3)", ha='center', color=DARK, fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig(f"{FIGS}/mapping_classes.png", bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("[OK] mapping_classes.png")

# ─────────────────────────────────────────────
# FIG 2 : EXTRACTION FEATURES 113D
# ─────────────────────────────────────────────
def fig_features_113d():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.text(7, 5.7, "Extraction des 113 caractéristiques physiques depuis une ROI",
            ha='center', fontsize=13, fontweight='bold', color=DARK)

    # ROI box
    roi = FancyBboxPatch((0.2, 1.5), 1.8, 3.0, boxstyle="round,pad=0.1",
                          fc="#E3F2FD", ec=SONATRACH_BLUE, lw=2)
    ax.add_patch(roi)
    ax.text(1.1, 3.1, "ROI\nTorchère", ha='center', va='center', color=SONATRACH_BLUE,
            fontsize=11, fontweight='bold')
    ax.text(1.1, 2.5, "224×224 px", ha='center', color=GRAY, fontsize=9)

    features = [
        ("Histogramme\nHSV", "94 dim", "#E8F5E9", SONATRACH_GREEN, 3.8),
        ("Texture\nLBP", "10 dim", "#FFF3E0", ORANGE, 5.1),
        ("Matrice\nGLCM", "4 dim", "#F3E5F5", "#7B1FA2", 6.4),
        ("Ratio\nFumée", "1 dim", "#FFEBEE", RED, 7.7),
        ("Intensité\nRGB", "3 dim", "#E0F2F1", "#00796B", 9.0),
        ("Pixels\nSombres", "1 dim", "#FBE9E7", "#BF360C", 10.3),
    ]

    ax.annotate('', xy=(3.2, 3.0), xytext=(2.0, 3.0),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))

    for name, dim, bg, fg, x in features:
        box = FancyBboxPatch((x-0.5, 1.8), 1.1, 2.4, boxstyle="round,pad=0.05",
                              fc=bg, ec=fg, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 0.05, 3.1, name, ha='center', va='center', color=DARK, fontsize=8.5, fontweight='bold')
        ax.text(x + 0.05, 2.15, dim, ha='center', va='center', color=fg, fontsize=9, fontweight='bold')

    # Output vector
    vec = FancyBboxPatch((11.8, 2.2), 1.9, 1.6, boxstyle="round,pad=0.08",
                          fc=SONATRACH_BLUE, ec='white', lw=2)
    ax.add_patch(vec)
    ax.text(12.75, 3.2, "Vecteur", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    ax.text(12.75, 2.75, "113 dim", ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    ax.annotate('', xy=(11.8, 3.0), xytext=(10.9, 3.0),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    # Bracket under features
    ax.annotate('', xy=(10.8, 1.5), xytext=(3.2, 1.5),
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=1))
    ax.text(7.0, 1.2, "→ Concaténation →", ha='center', color=GRAY, fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(f"{FIGS}/features_113d.png", bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("[OK] features_113d.png")

# ─────────────────────────────────────────────
# FIG 3 : DISTRIBUTION DU DATASET (ROI)
# ─────────────────────────────────────────────
def fig_dataset_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle("Distribution du jeu de données — ROI extraites (test3)", fontsize=13, fontweight='bold', color=DARK, y=1.02)

    # Données réelles
    classes_yolo = ['Dark-Flare', 'Dark-Smoke', 'Light-Flare', 'Light-Smoke', 'Medium-Flare', 'Medium-Smoke']
    colors_yolo = [RED, "#880E4F", SONATRACH_BLUE, SONATRACH_GREEN, ORANGE, YELLOW]

    # Test3 split : 267 bonne + 256 moyenne + 1096 mauvaise = 1619
    quality_counts = [267, 256, 1096]
    quality_labels = ['Bonne\n(267)', 'Moyenne\n(256)', 'Mauvaise\n(1 096)']
    quality_colors = [SONATRACH_GREEN, ORANGE, RED]

    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        quality_counts,
        labels=quality_labels,
        colors=quality_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.03, 0.03, 0.03),
        textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_color('white')
    axes[0].set_title("Répartition par qualité\n(test3 — 1 619 ROI)", fontsize=11, color=DARK)

    # Bar chart splits
    splits = ['Train (~7 500)', 'Validation (~1 200)', 'Test3 (1 619)']
    bonne   = [0.165, 0.165, 267/1619]
    moyenne = [0.158, 0.158, 256/1619]
    mauvaise= [0.677, 0.677, 1096/1619]

    x = np.arange(len(splits))
    width = 0.25
    bars1 = axes[1].bar(x - width, [b*100 for b in bonne],    width, label='Bonne',    color=SONATRACH_GREEN, alpha=0.85)
    bars2 = axes[1].bar(x,         [m*100 for m in moyenne],  width, label='Moyenne',  color=ORANGE, alpha=0.85)
    bars3 = axes[1].bar(x + width, [mv*100 for mv in mauvaise], width, label='Mauvaise', color=RED, alpha=0.85)

    axes[1].set_ylabel("Proportion (%)")
    axes[1].set_title("Répartition par split", fontsize=11, color=DARK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits, fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].set_facecolor('#F8F9FA')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_ylim(0, 85)

    plt.tight_layout()
    plt.savefig(f"{FIGS}/dataset_distribution.png", bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("[OK] dataset_distribution.png")

# ─────────────────────────────────────────────
# FIG 4 : TIMELINE YOLO
# ─────────────────────────────────────────────
def fig_yolo_timeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)

    ax.text(7, 4.7, "Évolution de l'architecture YOLO (2016–2026)", ha='center',
            fontsize=13, fontweight='bold', color=DARK)

    versions = [
        ("YOLOv1", "2016", "Redmon et al.", "mAP: 63.4%\n45 FPS", 1.0),
        ("YOLOv3", "2018", "Redmon et al.", "mAP: 57.9%\n3 échelles", 3.0),
        ("YOLOv5", "2020", "Ultralytics", "mAP: 56.8%\nOptimisation", 5.2),
        ("YOLOv8", "2023", "Ultralytics", "Anchor-free\nAPI unifiée", 7.4),
        ("YOLO11", "2024", "Ultralytics", "Plus précis\n✓ Choix retenu", 9.6),
        ("YOLO26", "2026", "Ultralytics", "NMS-free\nedge optimisé", 11.8),
    ]

    colors = [GRAY, GRAY, GRAY, GRAY, SONATRACH_BLUE, SONATRACH_GREEN]

    # Timeline line
    ax.plot([0.5, 13.5], [2.2, 2.2], color=GRAY, lw=2, zorder=1)

    for (name, year, author, desc, x), color in zip(versions, colors):
        lw = 3 if color == SONATRACH_BLUE else 1.5
        fc = "#E3F2FD" if color == SONATRACH_BLUE else "#F5F5F5"

        # Dot on timeline
        ax.plot(x, 2.2, 'o', color=color, markersize=14, zorder=3)
        ax.text(x, 2.2, name[4:], ha='center', va='center', color='white', fontsize=7, fontweight='bold')

        # Box above
        box = FancyBboxPatch((x-0.85, 2.6), 1.7, 1.8, boxstyle="round,pad=0.08",
                              fc=fc, ec=color, lw=lw)
        ax.add_patch(box)
        ax.text(x, 4.25, name, ha='center', va='center', color=color, fontsize=9, fontweight='bold')
        ax.text(x, 3.9, year, ha='center', va='center', color=GRAY, fontsize=8)
        ax.text(x, 3.6, author, ha='center', va='center', color=GRAY, fontsize=7.5, style='italic')
        for i, line in enumerate(desc.split('\n')):
            fc_txt = color if color == SONATRACH_BLUE else DARK
            ax.text(x, 3.15 - i*0.3, line, ha='center', va='center', color=fc_txt, fontsize=7.5)

        # Connector
        ax.plot([x, x], [2.6, 2.34], color=color, lw=1.5)

    plt.tight_layout()
    plt.savefig(f"{FIGS}/yolo_timeline.png", bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("[OK] yolo_timeline.png")

# ─────────────────────────────────────────────
# FIG 5 : PIPELINE GLOBAL (ARCHITECTURE SYSTÈME)
# ─────────────────────────────────────────────
def fig_pipeline_global():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)

    ax.text(8, 6.7, "Architecture Globale du Système de Surveillance Intelligent des Torchères",
            ha='center', fontsize=13, fontweight='bold', color=DARK)

    def box(ax, x, y, w, h, label, sublabel="", fc="#E3F2FD", ec=SONATRACH_BLUE, fs=10):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                            fc=fc, ec=ec, lw=2)
        ax.add_patch(b)
        ax.text(x, y + (0.12 if sublabel else 0), label, ha='center', va='center',
                color=DARK, fontsize=fs, fontweight='bold')
        if sublabel:
            ax.text(x, y-0.25, sublabel, ha='center', va='center', color=GRAY, fontsize=8)

    def arrow(ax, x1, y1, x2, y2, label="", color=DARK):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2 + 0.15
            ax.text(mx, my, label, ha='center', color=GRAY, fontsize=8, style='italic')

    # INPUT
    box(ax, 1.2, 3.5, 2.0, 1.2, "📹 Caméra\nRGB", "Flux vidéo", fc="#E8F5E9", ec=SONATRACH_GREEN)

    # MODULE A
    box(ax, 4.5, 5.2, 2.4, 1.0, "Prétraitement", "640×640 px", fc="#F3E5F5", ec="#7B1FA2")
    box(ax, 4.5, 3.5, 2.4, 1.2, "YOLO11m", "Détection objets\n6 classes", fc="#E3F2FD", ec=SONATRACH_BLUE)
    box(ax, 4.5, 1.8, 2.4, 1.0, "NMS", "Suppression\nnon-maxima", fc="#F3E5F5", ec="#7B1FA2")

    # ROI
    box(ax, 8.0, 3.5, 2.0, 1.2, "ROI\nExtraction", "Bboxes → crops", fc="#FFF3E0", ec=ORANGE)

    # MODULE B — deux branches
    box(ax, 11.2, 5.1, 2.4, 1.0, "Features 113D", "HSV+LBP+GLCM", fc="#E8F5E9", ec=SONATRACH_GREEN)
    box(ax, 11.2, 3.5, 2.4, 1.0, "SVM RBF", "C=10, γ=0.01", fc="#E8F5E9", ec=SONATRACH_GREEN)
    box(ax, 11.2, 1.9, 2.4, 1.0, "Frame-Level\nReasoning", "Dark-Smoke priority", fc="#FFF9C4", ec="#F9A825")

    # OUTPUT
    box(ax, 14.8, 5.1, 1.8, 0.8, "✅ Bonne", fc="#C8E6C9", ec=SONATRACH_GREEN, fs=9)
    box(ax, 14.8, 3.5, 1.8, 0.8, "⚠️ Moyenne", fc="#FFE0B2", ec=ORANGE, fs=9)
    box(ax, 14.8, 1.9, 1.8, 0.8, "🚨 Mauvaise", fc="#FFCDD2", ec=RED, fs=9)

    # Log
    box(ax, 14.8, 0.6, 1.8, 0.7, "Log CSV", "Horodaté", fc="#F5F5F5", ec=GRAY, fs=8)

    # Arrows
    arrow(ax, 2.2, 3.5, 3.3, 3.5)
    ax.annotate('', xy=(4.5, 4.7), xytext=(4.5, 4.1), arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    arrow(ax, 4.5, 2.9, 4.5, 2.3)
    arrow(ax, 5.7, 3.5, 7.0, 3.5, "Bboxes")
    arrow(ax, 9.0, 3.5, 10.0, 3.5)
    ax.annotate('', xy=(11.2, 4.6), xytext=(11.2, 4.0), arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    ax.annotate('', xy=(11.2, 3.0), xytext=(11.2, 2.4), arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    arrow(ax, 12.4, 5.1, 13.9, 5.1)
    arrow(ax, 12.4, 3.5, 13.9, 3.5)
    arrow(ax, 12.4, 1.9, 13.9, 1.9)
    ax.annotate('', xy=(14.8, 0.95), xytext=(14.8, 1.55), arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    # Module labels
    ax.text(4.5, 6.5, "MODULE A — Détection (YOLO11m)", ha='center', color=SONATRACH_BLUE,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec=SONATRACH_BLUE))
    ax.text(11.2, 6.5, "MODULE B — Classification (SVM)", ha='center', color=SONATRACH_GREEN,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec=SONATRACH_GREEN))
    ax.text(14.8, 6.5, "SORTIE", ha='center', color=DARK, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{FIGS}/pipeline_global.png", bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print("[OK] pipeline_global.png")

# ─────────────────────────────────────────────
# FIG 6 : COMPARAISON ARCHITECTURES CNN
# ─────────────────────────────────────────────
def fig_cnn_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle("Comparaison des architectures CNN candidates", fontsize=13, fontweight='bold', color=DARK)

    models = ['ResNet-50', 'VGG-16', 'EfficientNet-B0\n(choix retenu)', 'MobileNetV2', 'InceptionV3']
    params = [25.6, 138.0, 5.3, 3.4, 23.8]            # millions
    top1   = [76.1,  71.6, 77.1, 72.0, 77.9]          # ImageNet Top-1 %
    colors = [GRAY, GRAY, SONATRACH_BLUE, GRAY, GRAY]

    ax = axes[0]
    bars = ax.barh(models, params, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xlabel("Nombre de paramètres (millions)")
    ax.set_title("Complexité du modèle", fontsize=11)
    ax.set_facecolor('#F8F9FA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, params):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val}M', va='center', fontsize=9, color=DARK)

    ax2 = axes[1]
    bars2 = ax2.barh(models, top1, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_xlabel("ImageNet Top-1 Accuracy (%)")
    ax2.set_title("Performance pré-entraînement", fontsize=11)
    ax2.set_facecolor('#F8F9FA')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(65, 82)
    for bar, val in zip(bars2, top1):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val}%', va='center', fontsize=9, color=DARK)

    patch = mpatches.Patch(color=SONATRACH_BLUE, label='Architecture retenue')
    fig.legend(handles=[patch], loc='lower center', ncol=1, fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{FIGS}/cnn_comparison.png", bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("[OK] cnn_comparison.png")

# ─────────────────────────────────────────────
# FIG 7 : FINE-TUNING STRATEGY
# ─────────────────────────────────────────────
def fig_finetuning():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)

    ax.text(6.5, 4.7, "Stratégie de Fine-tuning — EfficientNet-B0",
            ha='center', fontsize=13, fontweight='bold', color=DARK)

    # Phase 1
    ax.text(2.5, 4.2, "Phase 1 — Entraînement de la tête (10 epochs)", ha='center',
            color=SONATRACH_BLUE, fontsize=10, fontweight='bold')
    blocks = [
        ("ImageNet\nBackbone", "#E0E0E0", GRAY, True),
        ("MBConv\nBloc 1–5", "#E0E0E0", GRAY, True),
        ("MBConv\nBloc 6–7", "#E0E0E0", GRAY, True),
        ("Global\nPooling", "#C8E6C9", SONATRACH_GREEN, False),
        ("Dense\n1280→3", "#C8E6C9", SONATRACH_GREEN, False),
    ]
    for i, (label, bg, ec, frozen) in enumerate(blocks):
        x = 0.4 + i * 1.0
        fb = FancyBboxPatch((x, 1.8), 0.85, 1.6, boxstyle="round,pad=0.05", fc=bg, ec=ec, lw=2)
        ax.add_patch(fb)
        ax.text(x+0.425, 2.65, label, ha='center', va='center', color=DARK, fontsize=8)
        if frozen:
            ax.text(x+0.425, 1.65, "🔒 Gelé", ha='center', color=GRAY, fontsize=7.5)
        else:
            ax.text(x+0.425, 1.65, "✏️ Entraîné", ha='center', color=SONATRACH_GREEN, fontsize=7.5)
    ax.text(0.8, 1.15, f"LR = 1e-3 | Loss = Cross-Entropy pondérée", ha='left', color=GRAY, fontsize=9)

    # Phase 2
    ax.text(9.5, 4.2, "Phase 2 — Fine-tuning (20 epochs)", ha='center',
            color=ORANGE, fontsize=10, fontweight='bold')
    blocks2 = [
        ("ImageNet\nBackbone", "#E0E0E0", GRAY, True),
        ("MBConv\nBloc 1–5", "#E0E0E0", GRAY, True),
        ("MBConv\nBloc 6–7", "#FFE0B2", ORANGE, False),
        ("Global\nPooling", "#FFE0B2", ORANGE, False),
        ("Dense\n1280→3", "#FFE0B2", ORANGE, False),
    ]
    for i, (label, bg, ec, frozen) in enumerate(blocks2):
        x = 7.4 + i * 1.0
        fb = FancyBboxPatch((x, 1.8), 0.85, 1.6, boxstyle="round,pad=0.05", fc=bg, ec=ec, lw=2)
        ax.add_patch(fb)
        ax.text(x+0.425, 2.65, label, ha='center', va='center', color=DARK, fontsize=8)
        if frozen:
            ax.text(x+0.425, 1.65, "🔒 Gelé", ha='center', color=GRAY, fontsize=7.5)
        else:
            ax.text(x+0.425, 1.65, "✏️ Entraîné", ha='center', color=ORANGE, fontsize=7.5)
    ax.text(7.8, 1.15, f"LR = 1e-4 | Meilleur val acc = 95.27% (epoch 27)", ha='left', color=GRAY, fontsize=9)

    ax.annotate('', xy=(7.1, 2.6), xytext=(5.6, 2.6),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    ax.text(6.35, 2.85, "Déblocage\nblocs 6–7", ha='center', color=DARK, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIGS}/finetuning_strategy.png", bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("[OK] finetuning_strategy.png")

# ─────────────────────────────────────────────
# FIG 8 : QUALITÉ DE COMBUSTION (3 TYPES)
# ─────────────────────────────────────────────
def fig_combustion_types():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle("Les 3 niveaux de qualité de combustion", fontsize=13,
                 fontweight='bold', color='white', y=1.02)

    configs = [
        ("BONNE combustion", ["#FFF9C4", "#FFD54F", "#FF8F00", "#FFCC02"],
         SONATRACH_GREEN, "Flamme claire (jaune-blanc)\nLight-Flare / Light-Smoke\nEfficacité > 98%"),
        ("MOYENNE combustion", ["#FF8F00", "#E65100", "#BF360C", "#FF6D00"],
         ORANGE, "Flamme orange sombre\nMedium-Flare / Medium-Smoke\nEfficacité 80-98%"),
        ("MAUVAISE combustion", ["#37474F", "#263238", "#1a1a1a", "#4A148C"],
         RED, "Fumée noire dense\nDark-Flare / Dark-Smoke\nEfficacité < 80%"),
    ]

    for ax, (title, colors, border_color, desc) in zip(axes, configs):
        ax.set_facecolor('#1a1a1a')
        ax.set_xlim(0, 10); ax.set_ylim(0, 12)
        ax.axis('off')

        # Simulate flame/smoke with gradient patches
        np.random.seed(42)
        for _ in range(300):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 10)
            c = colors[np.random.randint(0, len(colors))]
            size = np.random.uniform(800, 3000)
            alpha = np.random.uniform(0.3, 0.85)
            ax.scatter(x, y, c=c, s=size, alpha=alpha, zorder=2)

        # Border
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, edgecolor=border_color, lw=4)
        ax.add_patch(rect)

        ax.set_title(title, color=border_color, fontsize=11, fontweight='bold', pad=8)
        ax.text(5, -0.5, desc, ha='center', va='top', color='white',
                fontsize=8.5, transform=ax.transData)

    plt.tight_layout()
    plt.savefig(f"{FIGS}/combustion_types.png", bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print("[OK] combustion_types.png")


def resolve_active_detection_run() -> Path | None:
    """Retourne le run de détection à utiliser pour copier les artefacts YOLO."""
    for run_dir in DETECTION_RUN_CANDIDATES:
        if (run_dir / "results.png").exists() or (run_dir / "results.csv").exists():
            return run_dir
    return None


def copy_detection_artifacts():
    """Copie les figures YOLO directement depuis le run actif vers le dossier du mémoire."""
    run_dir = resolve_active_detection_run()
    if run_dir is None:
        print("[WARN] Aucun run de détection trouvé (ni YOLO11m ni runs YOLOv8).")
        return

    assets = [
        ("results.png", "results.png"),
        ("results.png", "yolo_training_curves.png"),
        ("BoxF1_curve.png", "BoxF1_curve.png"),
        ("BoxPR_curve.png", "BoxPR_curve.png"),
        ("BoxP_curve.png", "BoxP_curve.png"),
        ("BoxR_curve.png", "BoxR_curve.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
        ("train_batch0.jpg", "train_batch0.jpg"),
        ("train_batch1.jpg", "train_batch1.jpg"),
        ("train_batch2.jpg", "train_batch2.jpg"),
        ("val_batch0_labels.jpg", "val_batch0_labels.jpg"),
        ("val_batch0_pred.jpg", "val_batch0_pred.jpg"),
        ("val_batch1_labels.jpg", "val_batch1_labels.jpg"),
        ("val_batch1_pred.jpg", "val_batch1_pred.jpg"),
        ("val_batch2_labels.jpg", "val_batch2_labels.jpg"),
        ("val_batch2_pred.jpg", "val_batch2_pred.jpg"),
    ]

    copied = 0
    for src_name, dst_name in assets:
        src = run_dir / src_name
        if not src.exists():
            continue
        shutil.copy2(src, FIGS_DIR / dst_name)
        copied += 1

    print(f"[OK] Artefacts YOLO copiés depuis : {run_dir.name} ({copied} fichier(s))")

# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Génération des figures vers : {FIGS_DIR}\n")
    fig_mapping_classes()
    fig_features_113d()
    fig_dataset_distribution()
    fig_yolo_timeline()
    fig_pipeline_global()
    fig_cnn_comparison()
    fig_finetuning()
    fig_combustion_types()
    copy_detection_artifacts()

    print(f"\n✅ Terminé ! {len(list(FIGS_DIR.iterdir()))} fichiers dans {FIGS_DIR}")
