# setup_venv.ps1
# ================
# Creates a .venv in the project folder and installs all dependencies.
# Run once from the project root:
#   .\setup_venv.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Korean Movie Review Rating Predictor -- Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Create virtual environment
if (-Not (Test-Path ".\.venv")) {
    Write-Host ""
    Write-Host "[1/4] Creating .venv ..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "  OK: .venv created" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "[1/4] .venv already exists -- skipping creation" -ForegroundColor DarkGray
}

# 2. Activate
Write-Host ""
Write-Host "[2/4] Activating .venv ..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 3. Install general dependencies
Write-Host ""
Write-Host "[3/4] Installing dependencies from requirements.txt (no C: drive cache) ..." -ForegroundColor Yellow
pip install --upgrade pip -q --no-cache-dir
pip install -r requirements.txt --no-cache-dir

# 4. Force-install CUDA version of PyTorch for RTX 4080 (CUDA 12.4)
Write-Host ""
Write-Host "[4/4] Installing PyTorch with CUDA 12.4 for RTX GPU support ..." -ForegroundColor Yellow
Write-Host "      (This is ~2.5 GB -- files go to .venv only, not C: drive)" -ForegroundColor DarkGray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall --no-cache-dir


Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup complete! Your .venv is ready." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate in future sessions:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run the pipeline:" -ForegroundColor Cyan
Write-Host "  python prepare_data.py"
Write-Host "  python train_model.py --mode full"
Write-Host "  python evaluate_model.py"
Write-Host "  python predict.py"
Write-Host ""
