# PFE - Detection de Torchere et Evaluation de la Qualite de Combustion

**Projet de Fin d'Etudes - Master Informatique**
**Specialite : Vision par Ordinateur & Intelligence Artificielle**
**Realise en monome par Rayan Yanis Tahraoui**
**Realise en collaboration avec Sonatrach - Direction R&D, Boumerdes, Algerie**

> Ce depot est republie dans le cadre de mon portfolio personnel. Le PFE a ete
> realise en monome par **Rayan** ; j'ai contribue en tant qu'aide exterieure
> sur le module de classification de la qualite de combustion — voir la section
> [Auteur & Contribution externe](#auteur--contribution-externe) ci-dessous
> pour le detail.
>
> **Ma contribution : le module de classification de la qualite de
> combustion** (`src/classification/`) — extraction des ROI, feature engineering
> (113 dimensions : histogrammes HSV, texture LBP/GLCM, ratios de fumee),
> entrainement et comparaison de trois classifieurs (SVM, CNN EfficientNet-B0,
> approche hybride CNN+SVM), et etude d'ablation des features. Le reste du
> projet (detection YOLO, pipeline temps reel, interface graphique) est
> l'oeuvre de Rayan.

---

## Contexte

Dans l'industrie petroliere et gaziere, les torcheres (gas flares) sont utilisees pour bruler les gaz excedentaires. Une combustion inefficace produit de la fumee noire et des emissions polluantes. Ce projet vise a developper un systeme intelligent de surveillance automatique des torcheres a partir d'images RGB et de videos.

---

## Objectifs

1. Detecter automatiquement les torcheres dans des images RGB ou des videos.
2. Localiser les flammes via un modele de detection d'objets (YOLO11m).
3. Analyser visuellement la combustion (couleur, fumee, intensite).
4. Evaluer et classifier la qualite de combustion : bonne / moyenne / mauvaise combustion.
5. Evaluer les modeles avec mAP, Precision, Recall, F1-score.

---

## Structure du Projet

    FlareDetection-And-Classification/
    |-- README.md
    |-- requirements.txt
    |-- .gitignore
    |-- main.py                  # CLI : train / predict / evaluate (detection YOLO)
    |-- app_gui.py                # Interface graphique (PySide6)
    |-- setup.ps1                 # Installation automatisee (venv + dependances)
    |-- data/                     # Videos/images d'entree (non versionnees)
    |-- src/
    |   |-- realtime_monitor.py   # Pipeline temps reel (detection + tracking + classification)
    |   |-- flare_processor.py    # Wrapper du pipeline pour la GUI
    |   |-- models/               # Entrainement / inference YOLO11 (detection)
    |   |-- classification/       # Extraction ROI, features, SVM / CNN / hybride
    |   |-- evaluation/           # Metriques (mAP, P, R, F1)
    |   |-- dataset/              # Conversion et split des datasets Roboflow
    |   +-- utils/                # Configuration centralisee + helpers
    +-- outputs/
        |-- models/                # Poids entraines (YOLO11-m detection, YOLO11-n structure)
        |-- classification/        # Modeles SVM / CNN / hybride entraines
        +-- demo/                  # Videos de demonstration annotees

---

## Installation

    git clone https://github.com/nassim-touat/FlareDetection-And-Classification.git
    cd FlareDetection-And-Classification
    powershell -ExecutionPolicy Bypass -File setup.ps1

Ou installation manuelle :

    python -m venv .venv
    .venv\Scripts\pip.exe install -r requirements.txt

---

## Auteur & Contribution externe

Ce PFE a ete realise **en monome par Rayan**, qui reste l'auteur et le
responsable academique de l'ensemble du projet. J'ai contribue de maniere
ponctuelle, en tant qu'aide exterieure, sur le module de classification.

| | Role | Perimetre |
|---|---|---|
| **Rayan Yanis Tahraoui** | Auteur du PFE (monome) | Ensemble du projet : detection YOLO11 (entrainement, inference, filtre structure), pipeline temps reel, interface graphique — voir [`src/models/`](src/models/), [`src/realtime_monitor.py`](src/realtime_monitor.py), [`app_gui.py`](app_gui.py) |
| **Nassim Touat** | Contributeur externe | Aide apportee sur le module de classification de la qualite de combustion : extraction des ROI (`extract_rois.py`), feature engineering 113-dim (`features.py`), entrainement/comparaison SVM / CNN / hybride, etude d'ablation — voir [`src/classification/`](src/classification/) |

**Partenaire industriel :** Sonatrach - Direction R&D, Boumerdes, Algerie
