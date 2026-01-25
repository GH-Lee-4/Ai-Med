# ML-Based Symptoms Diagnosis Training Guide

This system now uses **Machine Learning models** trained on medical datasets instead of rule-based pattern matching.

## Overview

The ML-based diagnosis system:
- Uses **TF-IDF vectorization** to convert symptom text into numerical features
- Trains **Random Forest** model for urgency classification
- Trains **Gradient Boosting** model for cause classification
- Learns patterns from training data to make predictions
- Can be trained on custom datasets

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `scikit-learn` - Machine learning library
- `pandas` - Data handling
- Other existing dependencies

### 2. Train the Models

**Option A: Use Default Training Data**
```bash
python train_models.py
```

**Option B: Use Custom Dataset**
```bash
python train_models.py your_dataset.json
```

### 3. Use the Trained Models

The models are automatically loaded when you use the diagnosis function:

```python
from backend.symptoms_diagnosis import diagnose_symptoms

result = diagnose_symptoms(
    symptoms_text="I have chest pain and shortness of breath",
    patient_age=45
)

print(result)
```

## Dataset Format

Your custom dataset should be a JSON file with the following structure:

```json
[
    {
        "symptoms": "chest pain pressure tightness",
        "urgency": "CRITICAL",
        "causes": ["Heart attack or angina", "Pulmonary embolism"],
        "risks": ["Risk of heart attack or cardiac event"]
    },
    {
        "symptoms": "mild headache fatigue",
        "urgency": "LOW",
        "causes": ["Tension headache", "Sleep deprivation"],
        "risks": ["General health monitoring recommended"]
    }
]
```

### Required Fields:
- **symptoms**: Text description of symptoms (string)
- **urgency**: One of "CRITICAL", "HIGH", "MODERATE", "LOW", "ROUTINE" (string)
- **causes**: List of possible causes (array of strings)
- **risks**: List of possible risks (array of strings)

### Example Dataset

See `example_dataset.json` for a sample dataset with 10 examples.

## Model Architecture

### Feature Extraction
- **TF-IDF Vectorization**: Converts symptom text into numerical features
- **N-grams**: Uses 1-3 word combinations to capture context
- **Max Features**: 5000 most important features

### Models
1. **Urgency Classifier**: Random Forest (200 trees, max depth 20)
   - Predicts urgency level from symptoms
   
2. **Cause Classifier**: Gradient Boosting (150 trees, learning rate 0.1)
   - Predicts primary cause and top 5 possible causes

### Model Persistence
- Models are saved in `models/diagnosis_models/` directory
- Automatically loaded when needed
- Can be retrained with new data

## Training Process

When you run `train_models.py`:

1. **Data Preparation**: Loads and validates training data
2. **Feature Extraction**: Converts text to TF-IDF vectors
3. **Data Splitting**: 80% training, 20% testing
4. **Model Training**: Trains both urgency and cause models
5. **Evaluation**: Calculates accuracy on test set
6. **Saving**: Saves models, vectorizer, and encoders

## Adding More Training Data

### Method 1: Expand Default Dataset
Edit `backend/ml_diagnosis.py` and add more examples to `_load_default_training_data()`

### Method 2: Use Custom JSON File
Create a JSON file with your data and train:
```bash
python train_models.py my_custom_data.json
```

### Method 3: Programmatic Training
```python
from backend.ml_diagnosis import train_models

custom_data = [
    {
        "symptoms": "your symptom text here",
        "urgency": "MODERATE",
        "causes": ["Possible cause 1", "Possible cause 2"],
        "risks": ["Risk 1", "Risk 2"]
    }
]

results = train_models(custom_data)
print(f"Accuracy: {results['urgency_accuracy']}")
```

## Model Performance

The default training data includes:
- **200+ training examples** with variations
- **Multiple symptom categories**: Cardiovascular, Neurological, Respiratory, Gastrointestinal, Urinary, Skin, Musculoskeletal, General, Mental Health
- **Balanced urgency levels**

Expected performance:
- **Urgency Accuracy**: ~85-95% (depending on data quality)
- **Cause Accuracy**: ~70-85% (cause prediction is more complex)

## Improving Model Performance

1. **Add More Training Data**: More examples = better generalization
2. **Balance Classes**: Ensure all urgency levels are well-represented
3. **Quality Data**: Use accurate, medically-relevant examples
4. **Feature Engineering**: Adjust TF-IDF parameters in `ml_diagnosis.py`
5. **Model Tuning**: Adjust hyperparameters (n_estimators, max_depth, etc.)

## Integration with Streamlit App

The Streamlit app automatically uses ML models if they're trained. If models aren't found, it falls back to rule-based diagnosis.

To ensure ML models are used:
1. Train models first: `python train_models.py`
2. Models will be automatically loaded when the app starts
3. Diagnosis will use ML predictions

## Troubleshooting

### "Models not trained" Error
**Solution**: Run `python train_models.py` first

### Low Accuracy
**Solution**: 
- Add more training data
- Ensure data quality is good
- Check for class imbalance

### Import Errors
**Solution**: 
```bash
pip install scikit-learn pandas
```

### Memory Issues
**Solution**: Reduce `max_features` in TF-IDF vectorizer or use smaller dataset

## Advanced Usage

### Retraining with New Data
```python
from backend.ml_diagnosis import get_ml_model

model = get_ml_model()
# Add new data to existing training data
new_data = [...]
model.training_data.extend(new_data)
model.train()
```

### Custom Model Parameters
Edit `backend/ml_diagnosis.py` to adjust:
- `max_features` in TfidfVectorizer
- `n_estimators`, `max_depth` in RandomForestClassifier
- `n_estimators`, `learning_rate` in GradientBoostingClassifier

## Notes

- **Medical Disclaimer**: This is for educational purposes only. Always consult healthcare professionals for medical advice.
- **Model Limitations**: ML models learn from training data. They may not handle rare or novel symptoms well.
- **Continuous Learning**: Retrain models periodically with new data to improve performance.

