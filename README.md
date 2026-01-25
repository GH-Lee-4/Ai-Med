# AI Medical Diagnosis System

A comprehensive medical diagnosis agent powered by artificial intelligence, integrating computer vision for medical image analysis and large language models for symptom diagnosis.

## Features

- **Multi-Modal Analysis**:
  - **Image Processing**: Filter application (Grayscale, Infrared, X-ray effects).
  - **Medical Image Classification**: Detects Pneumonia/TB, Skin Lesions, Bone Fractures, and more.
  - **Segmentation**: Highlights problem areas in images.
  - **Risk Assessment**: Calculates risk scores based on visual and diagnostic data.
- **Symptom Diagnosis**:
  - **ML-Based**: Uses Random Forest and Gradient Boosting models trained on medical datasets.
  - **AI Chat Assistant**: Interactive diagnosis using LLMs (requires API Key).

## Installation

1. **Install Python**: Ensure you have Python 3.9+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `cv2`, you may need to install `opencv-python-headless` or ensuring system dependencies are met.*

## Setup

### OpenAI/DashScope API Key (Optional)
To use the AI Chat Assistant features, set your API key:

**Windows (PowerShell):**
```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-api-key-here", "User")
```
*Restart your terminal after setting the variable.*

## Running the Application

This application is built with Streamlit. To run it:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Deployment

### Real Application (Streamlit Cloud - RECOMMENDED)
To host the functional AI application:
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect your GitHub repository.
3. Set your `OPENAI_API_KEY` in the Streamlit "Secrets" settings.

### Mockup Site (Vercel)
The root of this repository is configured to serve the `frontend/` directory on Vercel. Note that this is a **static mockup** and does not process AI data.

## Verified Status
