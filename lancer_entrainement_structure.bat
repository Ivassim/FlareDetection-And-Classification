@echo off
rem ─────────────────────────────────────────────────────────────────────
rem  Entraînement du modèle STRUCTURE (YOLO11-n, flare stack / chimney)
rem  Durée estimée : ~4-6 h sur RTX 3060 Laptop. Laisser tourner.
rem  Reprise apres interruption : ajouter resume=True (voir README_STRUCTURE.md)
rem ─────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
echo [INFO] Entrainement du modele structure (YOLO11-n)...
echo [INFO] Sortie : outputs\models\flare_structure_yolo11n_v1\
".venv\Scripts\python.exe" src\models\train_structure.py
if errorlevel 1 (
    echo.
    echo [ERREUR] L'entrainement s'est arrete avec une erreur. Voir messages ci-dessus.
    pause
) else (
    echo.
    echo [OK] Entrainement termine. Etape suivante :
    echo      double-cliquer  evaluer_structure.bat
    pause
)
