# Git Setup and Push Script for AI Medical Diagnosis Project
# PowerShell version

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Git Setup and Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "[OK] Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Or add Git to your system PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[1/6] Initializing git repository..." -ForegroundColor Yellow
git init

Write-Host ""
Write-Host "[2/6] Adding remote repository..." -ForegroundColor Yellow
try {
    git remote add origin https://github.com/GH-Lee-4/Ai-Med.git
    Write-Host "[OK] Remote added" -ForegroundColor Green
} catch {
    git remote set-url origin https://github.com/GH-Lee-4/Ai-Med.git
    Write-Host "[OK] Remote updated" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/6] Creating .gitignore file..." -ForegroundColor Yellow
if (-not (Test-Path .gitignore)) {
    $gitignoreContent = @"
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
ENV/
.env
.venv
*.pth
*.pkl
models/diagnosis_models/*.pkl
vector_store/
.streamlit/
test_report.json
test_summary.txt
*.log
.DS_Store
Thumbs.db
"@
    Set-Content -Path .gitignore -Value $gitignoreContent
    Write-Host "[OK] .gitignore created" -ForegroundColor Green
} else {
    Write-Host "[OK] .gitignore already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4/6] Adding all files to staging..." -ForegroundColor Yellow
git add .

Write-Host ""
Write-Host "[5/6] Committing changes..." -ForegroundColor Yellow
git commit -m "Initial commit: AI Medical Diagnosis System with ML models, chat integration, and image processing"

Write-Host ""
Write-Host "[6/6] Pushing to origin/main..." -ForegroundColor Yellow
git branch -M main
git push -u origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Done! Your code has been pushed to GitHub" -ForegroundColor Green
Write-Host "Repository: https://github.com/GH-Lee-4/Ai-Med.git" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
