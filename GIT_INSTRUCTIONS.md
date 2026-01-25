# Git Setup and Push Instructions

## Quick Setup (Using the Batch Script)

1. **Make sure Git is installed:**
   - Download from: https://git-scm.com/download/win
   - Or if already installed, make sure it's in your system PATH

2. **Run the batch script:**
   ```bash
   git_setup_and_push.bat
   ```

## Manual Setup (If script doesn't work)

### Step 1: Initialize Git Repository
```bash
git init
```

### Step 2: Add Remote Repository
```bash
git remote add origin https://github.com/GH-Lee-4/Ai-Med.git
```

### Step 3: Create .gitignore
Create a `.gitignore` file with:
```
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
```

### Step 4: Add and Commit Files
```bash
git add .
git commit -m "Initial commit: AI Medical Diagnosis System with ML models, chat integration, and image processing"
```

### Step 5: Push to GitHub
```bash
git branch -M main
git push -u origin main
```

## If You Get Authentication Errors

You may need to authenticate with GitHub:

1. **Using Personal Access Token (Recommended):**
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate a new token with `repo` permissions
   - When prompted for password, use the token instead

2. **Or use GitHub CLI:**
   ```bash
   gh auth login
   ```

## Troubleshooting

- **"git is not recognized"**: Install Git or add it to PATH
- **"remote origin already exists"**: Run `git remote set-url origin https://github.com/GH-Lee-4/Ai-Med.git`
- **Authentication failed**: Use Personal Access Token instead of password
- **"branch main does not exist"**: Run `git branch -M main` first
