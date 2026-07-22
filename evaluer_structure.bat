@echo off
rem ─────────────────────────────────────────────────────────────────────
rem  Évaluation écart de domaine du modèle STRUCTURE sur nos vidéos.
rem  À lancer APRES lancer_entrainement_structure.bat
rem ─────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
".venv\Scripts\python.exe" eval_structure_domaine.py
pause
