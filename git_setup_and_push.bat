@echo off
REM Git Setup and Push Script for AI Medical Diagnosis Project
REM This script will initialize git, add remote, commit, and push to GitHub

echo ========================================
echo Git Setup and Push Script
echo ========================================
echo.

REM Check if git is available
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    echo Or add Git to your system PATH
    pause
    exit /b 1
)

echo [1/6] Initializing git repository...
git init

echo.
echo [2/6] Adding remote repository...
git remote add origin https://github.com/GH-Lee-4/Ai-Med.git
REM If remote already exists, update it
git remote set-url origin https://github.com/GH-Lee-4/Ai-Med.git

echo.
echo [3/6] Creating .gitignore file...
if not exist .gitignore (
    (
        echo __pycache__/
        echo *.pyc
        echo *.pyo
        echo *.pyd
        echo .Python
        echo env/
        echo venv/
        echo ENV/
        echo .env
        echo .venv
        echo *.pth
        echo *.pkl
        echo models/diagnosis_models/*.pkl
        echo vector_store/
        echo .streamlit/
        echo test_report.json
        echo test_summary.txt
        echo *.log
        echo .DS_Store
        echo Thumbs.db
    ) > .gitignore
    echo .gitignore created
) else (
    echo .gitignore already exists
)

echo.
echo [4/6] Adding all files to staging...
git add .

echo.
echo [5/6] Committing changes...
git commit -m "Initial commit: AI Medical Diagnosis System with ML models, chat integration, and image processing"

echo.
echo [6/6] Pushing to origin/main...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo Done! Your code has been pushed to GitHub
echo Repository: https://github.com/GH-Lee-4/Ai-Med.git
echo ========================================
pause
