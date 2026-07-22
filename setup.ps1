# Setup — PFE Sonatrach : Surveillance Intelligente des Torchères
# Ce script configure l'environnement complet en une fois.

Write-Host "=========================================="
Write-Host "  PFE Sonatrach — Setup de l'environnement"
Write-Host "=========================================="

# Etape 1 : venv
if (-Not (Test-Path ".venv")) {
    Write-Host "`n[1/4] Creation du venv..."
    python -m venv .venv
} else {
    Write-Host "`n[1/4] .venv existant detecte, on continue."
}

# Etape 2 : dependances
Write-Host "`n[2/4] Installation des dependances..."
.\.venv\Scripts\pip.exe install --upgrade pip -q
.\.venv\Scripts\pip.exe install -r requirements.txt

# Etape 3 : structure des dossiers
Write-Host "`n[3/4] Creation de la structure des dossiers..."
$dirs = @(
    "data/rois/train/bonne", "data/rois/train/moyenne", "data/rois/train/mauvaise",
    "data/rois/val/bonne",   "data/rois/val/moyenne",   "data/rois/val/mauvaise",
    "data/rois/test/bonne",  "data/rois/test/moyenne",  "data/rois/test/mauvaise",
    "data/rois/test2/bonne", "data/rois/test2/moyenne", "data/rois/test2/mauvaise",
    "data/rois/test3/bonne", "data/rois/test3/moyenne", "data/rois/test3/mauvaise",
    "outputs/realtime",
    "outputs/classification/visualizations"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "  Dossiers crees."

# Etape 4 : verification
Write-Host "`n[4/4] Verification de l'environnement..."
$checks = @{
    "YOLOv8 model (v3)"      = "outputs\models\gas_flare_yolov8m_v3\weights\best.pt"
    "SVM model"              = "outputs\classification\svm\svm_model.pkl"
    "CNN model"              = "outputs\classification\cnn\best_model.pt"
    "Hybrid SVM model"       = "outputs\classification\hybrid\hybrid_svm_model.pkl"
    "requirements.txt"       = "requirements.txt"
    "realtime_monitor.py"    = "src\realtime_monitor.py"
}

$allOk = $true
foreach ($label in $checks.Keys) {
    $path = $checks[$label]
    if (Test-Path $path) {
        $size = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "  [OK] $label ($size MB)"
    } else {
        Write-Host "  [MANQUANT] $label -> $path"
        $allOk = $false
    }
}

Write-Host ""
if ($allOk) {
    Write-Host "=========================================="
    Write-Host "  Setup termine ! Commandes disponibles :"
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "  # Pipeline temps reel sur une video :"
    Write-Host "  .\.venv\Scripts\python.exe src/realtime_monitor.py --source data/VOTRE_VIDEO.mp4"
    Write-Host ""
    Write-Host "  # Classification SVM (re-entrainement) :"
    Write-Host "  .\.venv\Scripts\python.exe -m src.classification.train_svm"
    Write-Host ""
    Write-Host "  # Visualisation des predictions :"
    Write-Host "  .\.venv\Scripts\python.exe -m src.classification.visualize_predictions"
} else {
    Write-Host "[ATTENTION] Certains fichiers sont manquants."
    Write-Host "Verifiez que vous avez bien clone le repo complet (git clone --depth=1)."
}
