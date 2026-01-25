"""
Machine Learning-based Symptoms Diagnosis Module
Uses trained models instead of rule-based pattern matching
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from enum import Enum


class UrgencyLevel(Enum):
    """Urgency levels for medical conditions"""
    CRITICAL = "Critical - Seek immediate emergency care"
    HIGH = "High - Consult a doctor within 24 hours"
    MODERATE = "Moderate - Schedule appointment within a few days"
    LOW = "Low - Monitor and consult if symptoms persist"
    ROUTINE = "Routine - Self-care may be sufficient"


class MLDiagnosisModel:
    """
    Machine Learning-based diagnosis model
    Trains on datasets and makes predictions based on learned patterns
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the ML diagnosis model"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Models for different predictions
        self.urgency_model = None
        self.cause_model = None
        self.risk_model = None
        
        # Feature extractors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Label encoders
        self.urgency_encoder = LabelEncoder()
        self.cause_encoder = LabelEncoder()
        
        # Training data storage
        self.training_data = []
        
    def _load_default_training_data(self) -> List[Dict]:
        """
        Generate comprehensive default training dataset
        This simulates a real medical dataset
        """
        training_data = []
        
        # Cardiovascular conditions
        training_data.extend([
            {
                "symptoms": "chest pain pressure tightness squeezing",
                "urgency": "CRITICAL",
                "causes": ["Heart attack or angina", "Pulmonary embolism", "Pneumonia"],
                "risks": ["Risk of heart attack or cardiac event", "Potential for life-threatening condition"]
            },
            {
                "symptoms": "chest pain after eating heartburn",
                "urgency": "MODERATE",
                "causes": ["Gastroesophageal reflux disease (GERD)", "Hiatal hernia"],
                "risks": ["Risk of esophageal damage"]
            },
            {
                "symptoms": "shortness of breath difficulty breathing can't breathe",
                "urgency": "CRITICAL",
                "causes": ["Asthma", "Pneumonia", "Heart failure", "Pulmonary embolism"],
                "risks": ["Risk of respiratory failure", "Potential for severe asthma attack"]
            },
            {
                "symptoms": "palpitations racing heart irregular heartbeat",
                "urgency": "HIGH",
                "causes": ["Anxiety or stress", "Arrhythmia", "Hyperthyroidism"],
                "risks": ["Risk of arrhythmia complications"]
            },
            {
                "symptoms": "swollen legs edema fluid retention",
                "urgency": "HIGH",
                "causes": ["Heart failure", "Kidney disease", "Deep vein thrombosis (DVT)"],
                "risks": ["Risk of deep vein thrombosis (DVT)", "Potential for heart or kidney failure"]
            },
        ])
        
        # Neurological conditions
        training_data.extend([
            {
                "symptoms": "severe headache worst headache thunderclap",
                "urgency": "HIGH",
                "causes": ["Migraine", "Meningitis", "Stroke (rare)", "Brain tumor (rare)"],
                "risks": ["Risk of stroke or brain hemorrhage", "Potential for meningitis"]
            },
            {
                "symptoms": "dizziness lightheaded vertigo feeling faint",
                "urgency": "MODERATE",
                "causes": ["Inner ear problems", "Low blood pressure", "Dehydration", "Anemia"],
                "risks": ["Risk of falls and injury", "Potential for fainting"]
            },
            {
                "symptoms": "numbness tingling pins and needles loss of sensation",
                "urgency": "HIGH",
                "causes": ["Pinched nerve", "Diabetes", "Stroke", "Peripheral neuropathy"],
                "risks": ["Risk of permanent nerve damage", "Potential for stroke"]
            },
            {
                "symptoms": "seizure convulsion fit epilepsy",
                "urgency": "CRITICAL",
                "causes": ["Epilepsy", "Head injury", "Stroke", "Brain tumor"],
                "risks": ["Risk of injury during seizure", "Potential for status epilepticus"]
            },
            {
                "symptoms": "confusion disorientation memory loss can't think",
                "urgency": "CRITICAL",
                "causes": ["Dehydration", "Infection", "Stroke", "Medication side effects"],
                "risks": ["Risk of falls and accidents", "Potential for serious underlying condition"]
            },
            {
                "symptoms": "blurred vision double vision vision loss",
                "urgency": "HIGH",
                "causes": ["Refractive errors", "Cataracts", "Glaucoma", "Stroke"],
                "risks": ["Risk of permanent vision loss", "Potential for stroke"]
            },
        ])
        
        # Respiratory conditions
        training_data.extend([
            {
                "symptoms": "cough persistent cough dry cough wet cough",
                "urgency": "LOW",
                "causes": ["Common cold", "Flu", "Bronchitis", "Pneumonia", "Asthma"],
                "risks": ["Risk of respiratory complications", "Potential for pneumonia"]
            },
            {
                "symptoms": "wheezing whistling sound breathing sounds",
                "urgency": "HIGH",
                "causes": ["Asthma", "COPD", "Bronchitis", "Allergic reaction"],
                "risks": ["Risk of severe asthma attack", "Potential for respiratory failure"]
            },
            {
                "symptoms": "congestion stuffy nose nasal congestion runny nose",
                "urgency": "LOW",
                "causes": ["Common cold", "Allergies", "Sinusitis"],
                "risks": ["Risk of sinus infection"]
            },
            {
                "symptoms": "sore throat throat pain difficulty swallowing",
                "urgency": "MODERATE",
                "causes": ["Viral infection (cold, flu)", "Bacterial infection (strep throat)", "Allergies"],
                "risks": ["Risk of complications from strep throat"]
            },
        ])
        
        # Gastrointestinal conditions
        training_data.extend([
            {
                "symptoms": "abdominal pain stomach pain belly ache cramps",
                "urgency": "MODERATE",
                "causes": ["Gastroenteritis", "Appendicitis", "Gallstones", "IBS", "Food poisoning"],
                "risks": ["Risk of appendicitis requiring surgery", "Potential for gastrointestinal complications"]
            },
            {
                "symptoms": "nausea vomiting throwing up feeling sick",
                "urgency": "MODERATE",
                "causes": ["Gastroenteritis", "Food poisoning", "Pregnancy", "Migraine"],
                "risks": ["Risk of dehydration", "Potential for electrolyte imbalance"]
            },
            {
                "symptoms": "diarrhea loose stools watery stools frequent bowel movements",
                "urgency": "MODERATE",
                "causes": ["Viral infection", "Bacterial infection", "Food poisoning", "IBS"],
                "risks": ["Risk of dehydration", "Potential for electrolyte imbalance"]
            },
            {
                "symptoms": "constipation can't go hard stools infrequent bowel movements",
                "urgency": "LOW",
                "causes": ["Inadequate fiber intake", "Dehydration", "Lack of exercise", "Medication side effects"],
                "risks": ["Risk of bowel obstruction"]
            },
            {
                "symptoms": "blood in stool rectal bleeding bloody stool black stool",
                "urgency": "CRITICAL",
                "causes": ["Hemorrhoids", "Anal fissure", "Inflammatory bowel disease (IBD)", "Colon cancer"],
                "risks": ["Risk of serious gastrointestinal condition", "Potential for colon cancer"]
            },
        ])
        
        # Urinary conditions
        training_data.extend([
            {
                "symptoms": "frequent urination urinating often need to pee often",
                "urgency": "MODERATE",
                "causes": ["Urinary tract infection (UTI)", "Diabetes", "Overactive bladder", "Pregnancy"],
                "risks": ["Risk of underlying diabetes"]
            },
            {
                "symptoms": "painful urination burning when urinating urination pain",
                "urgency": "MODERATE",
                "causes": ["Urinary tract infection (UTI)", "Bladder infection", "Kidney infection", "STI"],
                "risks": ["Risk of kidney infection", "Potential for sepsis"]
            },
            {
                "symptoms": "blood in urine bloody urine hematuria",
                "urgency": "CRITICAL",
                "causes": ["Urinary tract infection (UTI)", "Kidney stones", "Bladder cancer", "Kidney disease"],
                "risks": ["Risk of serious kidney or bladder condition", "Potential for cancer"]
            },
        ])
        
        # Skin conditions
        training_data.extend([
            {
                "symptoms": "rash skin irritation hives skin rash redness",
                "urgency": "LOW",
                "causes": ["Allergic reaction", "Eczema", "Psoriasis", "Contact dermatitis"],
                "risks": ["Risk of infection if scratched", "Potential for allergic reaction progression"]
            },
            {
                "symptoms": "itching itchy pruritus skin itching",
                "urgency": "LOW",
                "causes": ["Dry skin", "Eczema", "Allergic reaction", "Liver disease"],
                "risks": ["Risk of skin damage from scratching"]
            },
        ])
        
        # Musculoskeletal conditions
        training_data.extend([
            {
                "symptoms": "joint pain arthritis stiffness joint stiffness achy joints",
                "urgency": "LOW",
                "causes": ["Arthritis", "Osteoarthritis", "Rheumatoid arthritis", "Gout"],
                "risks": ["Risk of decreased mobility", "Potential for chronic arthritis"]
            },
            {
                "symptoms": "back pain lower back pain upper back pain spine pain",
                "urgency": "LOW",
                "causes": ["Muscle strain", "Herniated disc", "Osteoarthritis", "Poor posture"],
                "risks": ["Risk of chronic pain", "Potential for nerve damage"]
            },
        ])
        
        # General conditions
        training_data.extend([
            {
                "symptoms": "fever high temperature chills feeling hot sweating",
                "urgency": "MODERATE",
                "causes": ["Viral infection (flu, cold)", "Bacterial infection", "Urinary tract infection"],
                "risks": ["Risk of complications from infection", "Potential for sepsis in severe cases"]
            },
            {
                "symptoms": "fatigue tiredness exhaustion feeling tired low energy",
                "urgency": "LOW",
                "causes": ["Sleep deprivation", "Anemia", "Thyroid problems", "Depression", "Diabetes"],
                "risks": ["Risk of decreased quality of life", "Potential for underlying chronic condition"]
            },
            {
                "symptoms": "weight loss losing weight unintended weight loss",
                "urgency": "MODERATE",
                "causes": ["Hyperthyroidism", "Diabetes", "Cancer", "Depression", "Digestive disorders"],
                "risks": ["Risk of malnutrition", "Potential for underlying serious condition"]
            },
            {
                "symptoms": "excessive thirst very thirsty drinking a lot",
                "urgency": "MODERATE",
                "causes": ["Diabetes", "Dehydration", "Diabetes insipidus"],
                "risks": ["Risk of diabetes"]
            },
        ])
        
        # Mental health conditions
        training_data.extend([
            {
                "symptoms": "anxiety feeling anxious panic worry nervous",
                "urgency": "ROUTINE",
                "causes": ["Generalized anxiety disorder", "Panic disorder", "Stress"],
                "risks": ["Risk of panic attacks", "Potential for decreased quality of life"]
            },
            {
                "symptoms": "depression feeling sad hopeless down depressed",
                "urgency": "ROUTINE",
                "causes": ["Major depressive disorder", "Bipolar disorder", "Life events"],
                "risks": ["Risk of suicidal thoughts", "Potential for severe depression"]
            },
            {
                "symptoms": "insomnia can't sleep sleep problems trouble sleeping",
                "urgency": "ROUTINE",
                "causes": ["Stress", "Anxiety", "Depression", "Poor sleep habits"],
                "risks": ["Risk of fatigue and decreased function"]
            },
        ])
        
        # Add more variations and combinations
        additional_variations = []
        for entry in training_data:
            # Create variations with different wording
            variations = [
                entry["symptoms"] + " for several days",
                entry["symptoms"] + " suddenly",
                entry["symptoms"] + " with fever",
                entry["symptoms"] + " severe",
                "I have " + entry["symptoms"],
                "Experiencing " + entry["symptoms"],
            ]
            for var in variations:
                new_entry = entry.copy()
                new_entry["symptoms"] = var
                additional_variations.append(new_entry)
        
        training_data.extend(additional_variations)
        
        return training_data
    
    def prepare_training_data(self, custom_data: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for model training
        
        Args:
            custom_data: Optional custom training data
            
        Returns:
            X (features), y_urgency, y_causes, unique_causes
        """
        if custom_data:
            self.training_data = custom_data
        else:
            self.training_data = self._load_default_training_data()
        
        # Extract features and labels
        symptoms_texts = [entry["symptoms"] for entry in self.training_data]
        urgency_labels = [entry["urgency"] for entry in self.training_data]
        cause_labels = [entry["causes"][0] if entry["causes"] else "Unknown" for entry in self.training_data]
        
        # Vectorize symptoms text
        X = self.vectorizer.fit_transform(symptoms_texts)
        
        # Encode labels
        y_urgency = self.urgency_encoder.fit_transform(urgency_labels)
        y_causes = self.cause_encoder.fit_transform(cause_labels)
        
        # Get unique causes for reference
        unique_causes = list(self.cause_encoder.classes_)
        
        return X, y_urgency, y_causes, unique_causes
    
    def train(self, custom_data: Optional[List[Dict]] = None, test_size: float = 0.2):
        """
        Train the ML models
        
        Args:
            custom_data: Optional custom training data
            test_size: Proportion of data to use for testing
        """
        print("Preparing training data...")
        X, y_urgency, y_causes, unique_causes = self.prepare_training_data(custom_data)
        
        # Split data
        X_train, X_test, y_urgency_train, y_urgency_test = train_test_split(
            X, y_urgency, test_size=test_size, random_state=42, stratify=y_urgency
        )
        
        X_train_cause, X_test_cause, y_cause_train, y_cause_test = train_test_split(
            X, y_causes, test_size=test_size, random_state=42, stratify=y_causes
        )
        
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
        
        # Train urgency model (Random Forest for better interpretability)
        print("Training urgency classification model...")
        self.urgency_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.urgency_model.fit(X_train, y_urgency_train)
        urgency_pred = self.urgency_model.predict(X_test)
        urgency_acc = accuracy_score(y_urgency_test, urgency_pred)
        print(f"Urgency model accuracy: {urgency_acc:.3f}")
        
        # Train cause model (Gradient Boosting for better performance)
        print("Training cause classification model...")
        self.cause_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        self.cause_model.fit(X_train_cause, y_cause_train)
        cause_pred = self.cause_model.predict(X_test_cause)
        cause_acc = accuracy_score(y_cause_test, cause_pred)
        print(f"Cause model accuracy: {cause_acc:.3f}")
        
        # Save models
        self.save_models()
        
        print("Training completed!")
        return {
            "urgency_accuracy": float(urgency_acc),
            "cause_accuracy": float(cause_acc),
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        }
    
    def predict(self, symptoms_text: str, patient_age: Optional[int] = None) -> Dict:
        """
        Make diagnosis prediction using trained models
        
        Args:
            symptoms_text: Description of symptoms
            patient_age: Optional age of patient
            
        Returns:
            Dictionary with predictions
        """
        if self.urgency_model is None or self.cause_model is None:
            # Load models if not already loaded
            if not self.load_models():
                raise ValueError("Models not trained. Please train models first using train() method.")
        
        # Vectorize input
        symptoms_vector = self.vectorizer.transform([symptoms_text])
        
        # Predict urgency
        urgency_pred = self.urgency_model.predict(symptoms_vector)[0]
        urgency_proba = self.urgency_model.predict_proba(symptoms_vector)[0]
        urgency_label = self.urgency_encoder.inverse_transform([urgency_pred])[0]
        urgency_confidence = float(max(urgency_proba))
        
        # Predict cause
        cause_pred = self.cause_model.predict(symptoms_vector)[0]
        cause_proba = self.cause_model.predict_proba(symptoms_vector)[0]
        cause_label = self.cause_encoder.inverse_transform([cause_pred])[0]
        cause_confidence = float(max(cause_proba))
        
        # Get top causes
        top_cause_indices = np.argsort(cause_proba)[-5:][::-1]
        top_causes = [
            {
                "cause": self.cause_encoder.inverse_transform([idx])[0],
                "confidence": float(cause_proba[idx])
            }
            for idx in top_cause_indices
        ]
        
        # Map urgency to UrgencyLevel
        urgency_mapping = {
            "CRITICAL": UrgencyLevel.CRITICAL,
            "HIGH": UrgencyLevel.HIGH,
            "MODERATE": UrgencyLevel.MODERATE,
            "LOW": UrgencyLevel.LOW,
            "ROUTINE": UrgencyLevel.ROUTINE
        }
        urgency_level = urgency_mapping.get(urgency_label, UrgencyLevel.ROUTINE)
        
        # Generate risks based on urgency and cause
        risks = self._generate_risks(urgency_label, cause_label)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(urgency_level, patient_age)
        
        return {
            "urgency_level": urgency_level.value,
            "urgency_priority": urgency_label,
            "urgency_confidence": urgency_confidence,
            "primary_cause": cause_label,
            "cause_confidence": cause_confidence,
            "possible_causes": [c["cause"] for c in top_causes],
            "all_predictions": top_causes,
            "possible_risks": risks,
            "recommendations": recommendations,
            "matched_symptom_count": len(symptoms_text.split()),
            "explanations": self._generate_explanations(urgency_label, cause_label, risks)
        }
    
    def _generate_risks(self, urgency: str, cause: str) -> List[str]:
        """Generate risks based on urgency and cause"""
        risks = []
        
        if urgency == "CRITICAL":
            risks.extend([
                "Risk of life-threatening condition",
                "Potential for serious complications",
                "May require immediate medical intervention"
            ])
        elif urgency == "HIGH":
            risks.extend([
                "Risk of complications if untreated",
                "Potential for condition to worsen",
                "May require prompt medical attention"
            ])
        elif urgency == "MODERATE":
            risks.extend([
                "Risk of complications from delayed treatment",
                "Potential for condition progression",
                "May require medical evaluation"
            ])
        else:
            risks.extend([
                "General health monitoring recommended",
                "May require treatment if symptoms persist"
            ])
        
        return risks
    
    def _generate_recommendations(self, urgency: UrgencyLevel, patient_age: Optional[int] = None) -> List[str]:
        """Generate recommendations based on urgency"""
        recommendations = []
        
        if urgency == UrgencyLevel.CRITICAL:
            recommendations.append("🚨 Seek immediate emergency medical attention (call 911 or go to ER)")
            recommendations.append("Do not delay - this may be a life-threatening condition")
        elif urgency == UrgencyLevel.HIGH:
            recommendations.append("⚠️ Consult a healthcare provider within 24 hours")
            recommendations.append("Consider urgent care or emergency room if symptoms worsen")
        elif urgency == UrgencyLevel.MODERATE:
            recommendations.append("📅 Schedule an appointment with your doctor within a few days")
            recommendations.append("Monitor symptoms and seek immediate care if they worsen")
        elif urgency == UrgencyLevel.LOW:
            recommendations.append("👀 Monitor symptoms and maintain self-care")
            recommendations.append("Consult a doctor if symptoms persist or worsen")
        else:
            recommendations.append("💊 Self-care may be sufficient")
            recommendations.append("Consult a doctor if symptoms persist beyond a week")
        
        if patient_age:
            if patient_age < 18:
                recommendations.append("Note: Pediatric symptoms may require specialized care")
            elif patient_age > 65:
                recommendations.append("Note: Elderly patients may need closer monitoring")
        
        recommendations.append("Keep a symptom diary to track changes")
        recommendations.append("Stay hydrated and get adequate rest")
        recommendations.append("⚠️ This is not a substitute for professional medical advice")
        
        return recommendations
    
    def _generate_explanations(self, urgency: str, cause: str, risks: List[str]) -> Dict:
        """Generate explanations for the diagnosis"""
        return {
            "symptom_explanations": [],
            "cause_explanations": [{
                "term": cause,
                "explanation": f"{cause} is a possible cause that a doctor can help diagnose and treat."
            }],
            "risk_explanations": [{
                "term": risk,
                "explanation": f"{risk} is something to be aware of and discuss with your healthcare provider."
            } for risk in risks]
        }
    
    def save_models(self):
        """Save trained models to disk"""
        model_path = self.model_dir / "diagnosis_models"
        model_path.mkdir(exist_ok=True)
        
        # Save models
        with open(model_path / "urgency_model.pkl", "wb") as f:
            pickle.dump(self.urgency_model, f)
        
        with open(model_path / "cause_model.pkl", "wb") as f:
            pickle.dump(self.cause_model, f)
        
        # Save vectorizer
        with open(model_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save encoders
        with open(model_path / "urgency_encoder.pkl", "wb") as f:
            pickle.dump(self.urgency_encoder, f)
        
        with open(model_path / "cause_encoder.pkl", "wb") as f:
            pickle.dump(self.cause_encoder, f)
        
        print(f"Models saved to {model_path}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        model_path = self.model_dir / "diagnosis_models"
        
        try:
            # Load models
            with open(model_path / "urgency_model.pkl", "rb") as f:
                self.urgency_model = pickle.load(f)
            
            with open(model_path / "cause_model.pkl", "rb") as f:
                self.cause_model = pickle.load(f)
            
            # Load vectorizer
            with open(model_path / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load encoders
            with open(model_path / "urgency_encoder.pkl", "rb") as f:
                self.urgency_encoder = pickle.load(f)
            
            with open(model_path / "cause_encoder.pkl", "rb") as f:
                self.cause_encoder = pickle.load(f)
            
            print("Models loaded successfully")
            return True
        except FileNotFoundError:
            print("No saved models found. Please train models first.")
            return False


# Global model instance
_ml_model_instance = None


def get_ml_model() -> MLDiagnosisModel:
    """Get or create global ML model instance"""
    global _ml_model_instance
    if _ml_model_instance is None:
        _ml_model_instance = MLDiagnosisModel()
        # Try to load existing models
        _ml_model_instance.load_models()
    return _ml_model_instance


def diagnose_symptoms_ml(symptoms_text: str, patient_age: int = None, 
                        additional_info: str = None) -> Dict:
    """
    ML-based symptoms diagnosis function
    
    Args:
        symptoms_text: Description of symptoms
        patient_age: Age of patient (optional)
        additional_info: Additional medical information (optional)
    
    Returns:
        Dictionary containing diagnosis results
    """
    model = get_ml_model()
    
    # Combine symptoms and additional info
    full_text = symptoms_text
    if additional_info:
        full_text += " " + additional_info
    
    try:
        return model.predict(full_text, patient_age)
    except ValueError as e:
        # If models not trained, return error
        return {
            "error": str(e),
            "urgency_level": "Unable to determine",
            "possible_causes": [],
            "possible_risks": [],
            "recommendations": ["Please train the models first using train_models() function"]
        }


def train_models(custom_data: Optional[List[Dict]] = None) -> Dict:
    """
    Train the ML diagnosis models
    
    Args:
        custom_data: Optional custom training data (list of dicts with 'symptoms', 'urgency', 'causes', 'risks')
    
    Returns:
        Dictionary with training results
    """
    model = get_ml_model()
    return model.train(custom_data)

