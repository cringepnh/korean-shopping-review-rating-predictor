$ErrorActionPreference = "Stop"

Write-Host "Korean Shopping Review Rating Predictor -- Environment Setup" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath ".\.venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip --no-cache-dir
python -m pip install -r requirements.txt --no-cache-dir

Write-Host "Setup complete. Run:" -ForegroundColor Green
Write-Host "  python download_data.py"
Write-Host "  python prepare_data.py"
Write-Host "  python train_model.py"
Write-Host "  python evaluate_model.py"
