"""
Enhanced Machine Learning-based Symptoms Diagnosis Module
Improved accuracy with larger datasets, better models, and data augmentation
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from enum import Enum
import random


class UrgencyLevel(Enum):
    """Urgency levels for medical conditions"""
    CRITICAL = "Critical - Seek immediate emergency care"
    HIGH = "High - Consult a doctor within 24 hours"
    MODERATE = "Moderate - Schedule appointment within a few days"
    LOW = "Low - Monitor and consult if symptoms persist"
    ROUTINE = "Routine - Self-care may be sufficient"


class EnhancedMLDiagnosisModel:
    """
    Enhanced ML-based diagnosis model with improved accuracy
    Supports multiple datasets and advanced training techniques
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the enhanced ML diagnosis model"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Models for different predictions
        self.urgency_model = None
        self.cause_model = None
        
        # Enhanced feature extractors with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            ngram_range=(1, 4),  # Extended to 4-grams
            stop_words='english',
            min_df=1,  # Reduced to capture more features
            max_df=0.9,
            sublinear_tf=True,  # Apply sublinear tf scaling
            analyzer='word'
        )
        
        # Label encoders
        self.urgency_encoder = LabelEncoder()
        self.cause_encoder = LabelEncoder()
        
        # Training data storage
        self.training_data = []
        
    def _augment_data(self, data: List[Dict]) -> List[Dict]:
        """
        Data augmentation to increase training examples
        Creates variations of existing data
        """
        augmented = []
        
        for entry in data:
            symptoms = entry["symptoms"]
            
            # Add original
            augmented.append(entry)
            
            # Variations with different phrasings
            variations = [
                f"I have {symptoms}",
                f"I'm experiencing {symptoms}",
                f"Patient reports {symptoms}",
                f"Symptoms include {symptoms}",
                f"Complaining of {symptoms}",
                f"{symptoms} for the past few days",
                f"{symptoms} started suddenly",
                f"{symptoms} getting worse",
                f"{symptoms} with no improvement",
                f"Severe {symptoms}",
                f"Mild {symptoms}",
                f"Persistent {symptoms}",
                f"Recurring {symptoms}",
            ]
            
            for var in variations[:5]:  # Limit to avoid too much duplication
                new_entry = entry.copy()
                new_entry["symptoms"] = var
                augmented.append(new_entry)
        
        return augmented
    
    def _load_expanded_training_data(self) -> List[Dict]:
        """
        Generate comprehensive expanded training dataset
        Much larger and more diverse than default
        """
        training_data = []
        
        # ========== CARDIOVASCULAR CONDITIONS ==========
        cardiovascular_cases = [
            # Heart attack scenarios
            {"symptoms": "severe chest pain pressure tightness squeezing left arm pain", "urgency": "CRITICAL", "causes": ["Heart attack or angina"], "risks": ["Risk of heart attack or cardiac event"]},
            {"symptoms": "chest pain radiating to jaw neck shoulder", "urgency": "CRITICAL", "causes": ["Heart attack or angina"], "risks": ["Risk of heart attack or cardiac event"]},
            {"symptoms": "chest pain with sweating nausea shortness of breath", "urgency": "CRITICAL", "causes": ["Heart attack or angina"], "risks": ["Risk of heart attack or cardiac event"]},
            {"symptoms": "crushing chest pain unable to breathe", "urgency": "CRITICAL", "causes": ["Heart attack or angina"], "risks": ["Risk of heart attack or cardiac event"]},
            {"symptoms": "chest pain during exercise physical activity", "urgency": "HIGH", "causes": ["Heart attack or angina", "Exercise-induced angina"], "risks": ["Risk of heart attack or cardiac event"]},
            
            # Heart failure
            {"symptoms": "shortness of breath when lying down orthopnea", "urgency": "HIGH", "causes": ["Heart failure"], "risks": ["Risk of heart failure complications"]},
            {"symptoms": "shortness of breath with swollen ankles legs", "urgency": "HIGH", "causes": ["Heart failure"], "risks": ["Risk of heart failure complications"]},
            {"symptoms": "fatigue shortness of breath with activity", "urgency": "MODERATE", "causes": ["Heart failure", "Anemia"], "risks": ["Risk of heart failure complications"]},
            
            # Arrhythmias
            {"symptoms": "irregular heartbeat palpitations skipped beats", "urgency": "HIGH", "causes": ["Arrhythmia", "Atrial fibrillation"], "risks": ["Risk of arrhythmia complications"]},
            {"symptoms": "racing heart rate over 100 beats per minute", "urgency": "HIGH", "causes": ["Tachycardia", "Hyperthyroidism", "Anxiety"], "risks": ["Risk of arrhythmia complications"]},
            {"symptoms": "slow heart rate below 60 beats per minute", "urgency": "MODERATE", "causes": ["Bradycardia", "Medication side effects"], "risks": ["Risk of decreased cardiac output"]},
            
            # Other cardiovascular
            {"symptoms": "chest pain after eating spicy food", "urgency": "MODERATE", "causes": ["Gastroesophageal reflux disease (GERD)"], "risks": ["Risk of esophageal damage"]},
            {"symptoms": "chest pain when breathing deeply", "urgency": "MODERATE", "causes": ["Costochondritis", "Pleurisy"], "risks": ["Risk of underlying condition"]},
            {"symptoms": "swollen legs ankles pitting edema", "urgency": "HIGH", "causes": ["Heart failure", "Kidney disease", "Deep vein thrombosis (DVT)"], "risks": ["Risk of deep vein thrombosis (DVT)"]},
        ]
        training_data.extend(cardiovascular_cases)
        
        # ========== NEUROLOGICAL CONDITIONS ==========
        neurological_cases = [
            # Stroke symptoms
            {"symptoms": "sudden weakness numbness one side face arm leg", "urgency": "CRITICAL", "causes": ["Stroke", "Transient ischemic attack (TIA)"], "risks": ["Risk of permanent brain damage"]},
            {"symptoms": "sudden confusion trouble speaking understanding", "urgency": "CRITICAL", "causes": ["Stroke"], "risks": ["Risk of permanent brain damage"]},
            {"symptoms": "sudden severe headache worst ever experienced", "urgency": "CRITICAL", "causes": ["Stroke", "Brain hemorrhage", "Meningitis"], "risks": ["Risk of stroke or brain hemorrhage"]},
            {"symptoms": "sudden vision loss one eye both eyes", "urgency": "CRITICAL", "causes": ["Stroke", "Retinal detachment"], "risks": ["Risk of permanent vision loss"]},
            {"symptoms": "sudden trouble walking loss balance coordination", "urgency": "CRITICAL", "causes": ["Stroke"], "risks": ["Risk of permanent brain damage"]},
            
            # Migraine
            {"symptoms": "severe headache one side throbbing pulsating", "urgency": "HIGH", "causes": ["Migraine"], "risks": ["Risk of decreased quality of life"]},
            {"symptoms": "headache with nausea vomiting light sensitivity", "urgency": "HIGH", "causes": ["Migraine"], "risks": ["Risk of decreased quality of life"]},
            {"symptoms": "headache with visual aura flashing lights", "urgency": "HIGH", "causes": ["Migraine with aura"], "risks": ["Risk of decreased quality of life"]},
            
            # Seizures
            {"symptoms": "seizure convulsion shaking uncontrollable movements", "urgency": "CRITICAL", "causes": ["Epilepsy", "Brain injury", "Stroke"], "risks": ["Risk of injury during seizure"]},
            {"symptoms": "loss consciousness seizure activity", "urgency": "CRITICAL", "causes": ["Epilepsy", "Syncope"], "risks": ["Risk of injury during seizure"]},
            
            # Other neurological
            {"symptoms": "dizziness vertigo room spinning", "urgency": "MODERATE", "causes": ["Benign paroxysmal positional vertigo (BPPV)", "Meniere's disease"], "risks": ["Risk of falls and injury"]},
            {"symptoms": "numbness tingling hands feet", "urgency": "HIGH", "causes": ["Diabetes", "Peripheral neuropathy", "Vitamin B12 deficiency"], "risks": ["Risk of permanent nerve damage"]},
            {"symptoms": "memory loss confusion disorientation", "urgency": "CRITICAL", "causes": ["Dementia", "Delirium", "Infection"], "risks": ["Risk of falls and accidents"]},
            {"symptoms": "tremor shaking hands head", "urgency": "HIGH", "causes": ["Essential tremor", "Parkinson's disease"], "risks": ["Risk of decreased function"]},
        ]
        training_data.extend(neurological_cases)
        
        # ========== RESPIRATORY CONDITIONS ==========
        respiratory_cases = [
            # Asthma
            {"symptoms": "wheezing shortness of breath chest tightness", "urgency": "HIGH", "causes": ["Asthma"], "risks": ["Risk of severe asthma attack"]},
            {"symptoms": "difficulty breathing after exercise", "urgency": "MODERATE", "causes": ["Exercise-induced asthma"], "risks": ["Risk of severe asthma attack"]},
            {"symptoms": "coughing wheezing at night", "urgency": "MODERATE", "causes": ["Asthma", "Allergies"], "risks": ["Risk of severe asthma attack"]},
            
            # Pneumonia
            {"symptoms": "cough fever chills chest pain difficulty breathing", "urgency": "HIGH", "causes": ["Pneumonia"], "risks": ["Risk of respiratory complications"]},
            {"symptoms": "productive cough yellow green phlegm fever", "urgency": "HIGH", "causes": ["Pneumonia", "Bronchitis"], "risks": ["Risk of respiratory complications"]},
            {"symptoms": "shortness of breath with fever cough", "urgency": "HIGH", "causes": ["Pneumonia"], "risks": ["Risk of respiratory complications"]},
            
            # COPD
            {"symptoms": "chronic cough shortness of breath", "urgency": "MODERATE", "causes": ["Chronic obstructive pulmonary disease (COPD)"], "risks": ["Risk of respiratory failure"]},
            {"symptoms": "wheezing chronic bronchitis emphysema", "urgency": "MODERATE", "causes": ["Chronic obstructive pulmonary disease (COPD)"], "risks": ["Risk of respiratory failure"]},
            
            # Other respiratory
            {"symptoms": "sore throat difficulty swallowing", "urgency": "MODERATE", "causes": ["Strep throat", "Tonsillitis"], "risks": ["Risk of complications from strep throat"]},
            {"symptoms": "runny nose congestion sneezing", "urgency": "LOW", "causes": ["Common cold", "Allergies"], "risks": ["Risk of sinus infection"]},
            {"symptoms": "persistent cough more than three weeks", "urgency": "MODERATE", "causes": ["Bronchitis", "Post-nasal drip", "GERD"], "risks": ["Risk of underlying condition"]},
        ]
        training_data.extend(respiratory_cases)
        
        # ========== GASTROINTESTINAL CONDITIONS ==========
        gastrointestinal_cases = [
            # Appendicitis
            {"symptoms": "severe abdominal pain right lower quadrant", "urgency": "CRITICAL", "causes": ["Appendicitis"], "risks": ["Risk of appendicitis requiring surgery"]},
            {"symptoms": "abdominal pain with nausea vomiting fever", "urgency": "CRITICAL", "causes": ["Appendicitis"], "risks": ["Risk of appendicitis requiring surgery"]},
            
            # Gastroenteritis
            {"symptoms": "diarrhea vomiting nausea stomach cramps", "urgency": "MODERATE", "causes": ["Gastroenteritis", "Food poisoning"], "risks": ["Risk of dehydration"]},
            {"symptoms": "watery diarrhea multiple times per day", "urgency": "MODERATE", "causes": ["Gastroenteritis", "Viral infection"], "risks": ["Risk of dehydration"]},
            
            # GERD
            {"symptoms": "heartburn acid reflux after eating", "urgency": "LOW", "causes": ["Gastroesophageal reflux disease (GERD)"], "risks": ["Risk of esophageal damage"]},
            {"symptoms": "chest pain after eating lying down", "urgency": "MODERATE", "causes": ["Gastroesophageal reflux disease (GERD)"], "risks": ["Risk of esophageal damage"]},
            
            # Other GI
            {"symptoms": "blood in stool bright red", "urgency": "CRITICAL", "causes": ["Hemorrhoids", "Anal fissure", "Colon cancer"], "risks": ["Risk of serious gastrointestinal condition"]},
            {"symptoms": "black tarry stools melena", "urgency": "CRITICAL", "causes": ["Upper GI bleeding", "Peptic ulcer"], "risks": ["Risk of serious gastrointestinal condition"]},
            {"symptoms": "abdominal pain bloating gas", "urgency": "LOW", "causes": ["Irritable bowel syndrome (IBS)", "Food intolerance"], "risks": ["Risk of underlying digestive disorder"]},
            {"symptoms": "constipation hard stools infrequent bowel movements", "urgency": "LOW", "causes": ["Inadequate fiber intake", "Dehydration"], "risks": ["Risk of bowel obstruction"]},
        ]
        training_data.extend(gastrointestinal_cases)
        
        # ========== URINARY CONDITIONS ==========
        urinary_cases = [
            {"symptoms": "burning pain when urinating frequent urination", "urgency": "MODERATE", "causes": ["Urinary tract infection (UTI)"], "risks": ["Risk of kidney infection"]},
            {"symptoms": "urinary urgency frequency small amounts", "urgency": "MODERATE", "causes": ["Urinary tract infection (UTI)", "Overactive bladder"], "risks": ["Risk of kidney infection"]},
            {"symptoms": "blood in urine pink red brown urine", "urgency": "CRITICAL", "causes": ["Urinary tract infection (UTI)", "Kidney stones", "Bladder cancer"], "risks": ["Risk of serious kidney or bladder condition"]},
            {"symptoms": "flank pain back pain with fever chills", "urgency": "HIGH", "causes": ["Kidney infection", "Kidney stones"], "risks": ["Risk of sepsis"]},
            {"symptoms": "frequent urination excessive thirst", "urgency": "MODERATE", "causes": ["Diabetes"], "risks": ["Risk of diabetes complications"]},
        ]
        training_data.extend(urinary_cases)
        
        # ========== SKIN CONDITIONS ==========
        skin_cases = [
            {"symptoms": "rash red bumps itching", "urgency": "LOW", "causes": ["Allergic reaction", "Eczema"], "risks": ["Risk of infection if scratched"]},
            {"symptoms": "hives welts raised red bumps", "urgency": "MODERATE", "causes": ["Allergic reaction"], "risks": ["Potential for allergic reaction progression"]},
            {"symptoms": "skin rash with fever", "urgency": "MODERATE", "causes": ["Viral infection", "Bacterial infection"], "risks": ["Risk of underlying infection"]},
            {"symptoms": "dry itchy skin patches", "urgency": "LOW", "causes": ["Eczema", "Dry skin"], "risks": ["Risk of skin damage from scratching"]},
        ]
        training_data.extend(skin_cases)
        
        # ========== MUSCULOSKELETAL CONDITIONS ==========
        musculoskeletal_cases = [
            {"symptoms": "joint pain swelling stiffness", "urgency": "LOW", "causes": ["Arthritis", "Rheumatoid arthritis"], "risks": ["Risk of decreased mobility"]},
            {"symptoms": "back pain radiating down leg sciatica", "urgency": "MODERATE", "causes": ["Sciatica", "Herniated disc"], "risks": ["Risk of nerve damage"]},
            {"symptoms": "knee pain swelling after injury", "urgency": "MODERATE", "causes": ["Knee injury", "Torn ligament"], "risks": ["Risk of decreased mobility"]},
            {"symptoms": "neck pain stiffness limited movement", "urgency": "LOW", "causes": ["Muscle strain", "Poor posture"], "risks": ["Risk of chronic pain"]},
        ]
        training_data.extend(musculoskeletal_cases)
        
        # ========== GENERAL CONDITIONS ==========
        general_cases = [
            {"symptoms": "high fever over 103 degrees chills", "urgency": "HIGH", "causes": ["Bacterial infection", "Viral infection"], "risks": ["Risk of complications from infection"]},
            {"symptoms": "fatigue extreme tiredness no energy", "urgency": "LOW", "causes": ["Anemia", "Sleep deprivation", "Depression"], "risks": ["Risk of decreased quality of life"]},
            {"symptoms": "unexplained weight loss appetite loss", "urgency": "MODERATE", "causes": ["Cancer", "Hyperthyroidism", "Depression"], "risks": ["Risk of underlying serious condition"]},
            {"symptoms": "night sweats fever weight loss", "urgency": "MODERATE", "causes": ["Infection", "Cancer", "Hyperthyroidism"], "risks": ["Risk of underlying serious condition"]},
        ]
        training_data.extend(general_cases)
        
        # Apply data augmentation
        training_data = self._augment_data(training_data)
        
        return training_data
    
    def load_datasets(self, dataset_paths: List[str]) -> List[Dict]:
        """
        Load multiple dataset files and combine them
        
        Args:
            dataset_paths: List of paths to JSON dataset files
            
        Returns:
            Combined training data
        """
        all_data = []
        
        for path in dataset_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"Warning: Dataset file not found: {path}")
                continue
            
            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate format
                required_keys = ['symptoms', 'urgency', 'causes', 'risks']
                valid_data = []
                for i, entry in enumerate(data):
                    if all(key in entry for key in required_keys):
                        valid_data.append(entry)
                    else:
                        print(f"Warning: Skipping invalid entry {i} in {path}")
                
                all_data.extend(valid_data)
                print(f"Loaded {len(valid_data)} examples from {path}")
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        return all_data
    
    def prepare_training_data(self, custom_data: Optional[List[Dict]] = None, 
                             dataset_paths: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from multiple sources
        
        Args:
            custom_data: Direct training data
            dataset_paths: List of paths to dataset JSON files
            
        Returns:
            X (features), y_urgency, y_causes, unique_causes
        """
        self.training_data = []
        
        # Load default data
        default_data = self._load_expanded_training_data()
        self.training_data.extend(default_data)
        
        # Load from dataset files
        if dataset_paths:
            file_data = self.load_datasets(dataset_paths)
            self.training_data.extend(file_data)
        
        # Add custom data
        if custom_data:
            self.training_data.extend(custom_data)
        
        print(f"Total training examples: {len(self.training_data)}")
        
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
    
    def train(self, custom_data: Optional[List[Dict]] = None, 
              dataset_paths: Optional[List[str]] = None,
              test_size: float = 0.2,
              use_cross_validation: bool = True):
        """
        Train enhanced ML models with improved accuracy
        
        Args:
            custom_data: Optional custom training data
            dataset_paths: Optional list of dataset file paths
            test_size: Proportion of data to use for testing
            use_cross_validation: Whether to use cross-validation for evaluation
        """
        print("Preparing training data from multiple sources...")
        X, y_urgency, y_causes, unique_causes = self.prepare_training_data(custom_data, dataset_paths)
        
        # Split data with stratification if possible
        try:
            X_train, X_test, y_urgency_train, y_urgency_test = train_test_split(
                X, y_urgency, test_size=test_size, random_state=42, stratify=y_urgency
            )
        except ValueError:
            # If stratification fails, use random split
            X_train, X_test, y_urgency_train, y_urgency_test = train_test_split(
                X, y_urgency, test_size=test_size, random_state=42
            )
        
        # For causes, check if stratification is possible
        unique_causes_counts = np.bincount(y_causes)
        min_class_count = unique_causes_counts[unique_causes_counts > 0].min()
        
        if min_class_count >= 2:
            try:
                X_train_cause, X_test_cause, y_cause_train, y_cause_test = train_test_split(
                    X, y_causes, test_size=test_size, random_state=42, stratify=y_causes
                )
            except ValueError:
                X_train_cause, X_test_cause, y_cause_train, y_cause_test = train_test_split(
                    X, y_causes, test_size=test_size, random_state=42
                )
        else:
            # Use random split if some classes have too few examples
            X_train_cause, X_test_cause, y_cause_train, y_cause_test = train_test_split(
                X, y_causes, test_size=test_size, random_state=42
            )
        
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        # Train enhanced urgency model (Ensemble for better accuracy)
        print("\nTraining urgency classification model...")
        
        # Use ensemble of multiple models
        rf = RandomForestClassifier(
            n_estimators=300,  # Increased
            max_depth=25,  # Increased
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=12,
            learning_rate=0.05,
            random_state=42
        )
        
        # Use voting classifier for better accuracy
        self.urgency_model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            weights=[2, 1]
        )
        
        self.urgency_model.fit(X_train, y_urgency_train)
        
        # Evaluate
        if use_cross_validation:
            cv_scores = cross_val_score(self.urgency_model, X_train, y_urgency_train, cv=5, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        urgency_pred = self.urgency_model.predict(X_test)
        urgency_acc = accuracy_score(y_urgency_test, urgency_pred)
        urgency_f1 = f1_score(y_urgency_test, urgency_pred, average='weighted')
        print(f"Test accuracy: {urgency_acc:.3f}")
        print(f"Test F1-score: {urgency_f1:.3f}")
        
        # Train enhanced cause model
        print("\nTraining cause classification model...")
        
        self.cause_model = GradientBoostingClassifier(
            n_estimators=300,  # Increased
            max_depth=15,  # Increased
            learning_rate=0.05,  # Lower learning rate for better generalization
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        self.cause_model.fit(X_train_cause, y_cause_train)
        
        # Evaluate
        if use_cross_validation:
            cv_scores = cross_val_score(self.cause_model, X_train_cause, y_cause_train, cv=5, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        cause_pred = self.cause_model.predict(X_test_cause)
        cause_acc = accuracy_score(y_cause_test, cause_pred)
        cause_f1 = f1_score(y_cause_test, cause_pred, average='weighted')
        print(f"Test accuracy: {cause_acc:.3f}")
        print(f"Test F1-score: {cause_f1:.3f}")
        
        # Save models
        self.save_models()
        
        print("\n[SUCCESS] Training completed!")
        return {
            "urgency_accuracy": float(urgency_acc),
            "urgency_f1": float(urgency_f1),
            "cause_accuracy": float(cause_acc),
            "cause_f1": float(cause_f1),
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "total_features": X_train.shape[1]
        }
    
    def predict(self, symptoms_text: str, patient_age: Optional[int] = None) -> Dict:
        """Make diagnosis prediction using trained models"""
        if self.urgency_model is None or self.cause_model is None:
            if not self.load_models():
                print("Models not found. Attempting to train models automatically...")
                # Try to train with default datasets if available
                dataset_paths = []
                datasets_dir = Path("datasets")
                if datasets_dir.exists():
                    dataset_paths = [str(f) for f in datasets_dir.glob("*.json")]
                
                try:
                    self.train(dataset_paths=dataset_paths if dataset_paths else None)
                    print("Auto-training successful.")
                except Exception as e:
                    raise ValueError(f"Models not trained and auto-training failed: {e}")
        
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
        
        # Generate risks and recommendations
        risks = self._generate_risks(urgency_label, cause_label)
        recommendations = self._generate_recommendations(urgency_level, patient_age)
        
        # Generate explanations for all possible causes
        all_possible_causes = [c["cause"] for c in top_causes]
        explanations = self._generate_explanations(urgency_label, cause_label, risks, all_possible_causes)
        
        return {
            "urgency_level": urgency_level.value,
            "urgency_priority": urgency_label,
            "urgency_confidence": urgency_confidence,
            "primary_cause": cause_label,
            "cause_confidence": cause_confidence,
            "possible_causes": all_possible_causes,
            "all_predictions": top_causes,
            "possible_risks": risks,
            "recommendations": recommendations,
            "matched_symptom_count": len(symptoms_text.split()),
            "explanations": explanations
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
                "Potential for condition progression"
            ])
        else:
            risks.extend([
                "General health monitoring recommended"
            ])
        return risks
    
    def _generate_recommendations(self, urgency: UrgencyLevel, patient_age: Optional[int] = None) -> List[str]:
        """Generate recommendations based on urgency"""
        recommendations = []
        if urgency == UrgencyLevel.CRITICAL:
            recommendations.append("Seek immediate emergency medical attention (call 911 or go to ER)")
        elif urgency == UrgencyLevel.HIGH:
            recommendations.append("Consult a healthcare provider within 24 hours")
        elif urgency == UrgencyLevel.MODERATE:
            recommendations.append("Schedule an appointment with your doctor within a few days")
        elif urgency == UrgencyLevel.LOW:
            recommendations.append("Monitor symptoms and maintain self-care")
        else:
            recommendations.append("Self-care may be sufficient")
        
        if patient_age:
            if patient_age < 18:
                recommendations.append("Note: Pediatric symptoms may require specialized care")
            elif patient_age > 65:
                recommendations.append("Note: Elderly patients may need closer monitoring")
        
        recommendations.append("Keep a symptom diary to track changes")
        recommendations.append("This is not a substitute for professional medical advice")
        return recommendations
    
    def _get_cause_explanation(self, cause: str) -> str:
        """Get detailed explanation for a cause from the comprehensive database"""
        # Comprehensive cause explanations database
        cause_explanations_db = {
            # Cardiovascular causes
            "Heart attack or angina": "This happens when blood flow to your heart is blocked. It's a medical emergency. Angina is chest pain from reduced blood flow, while a heart attack means the heart muscle is being damaged.",
            "Pulmonary embolism": "This is when a blood clot blocks an artery in your lungs. It's serious and can be life-threatening, causing sudden shortness of breath and chest pain.",
            "Costochondritis": "This is inflammation of the cartilage that connects your ribs to your breastbone. It causes chest pain that can feel like a heart problem but is usually harmless.",
            "Pericarditis": "This is inflammation of the sac around your heart. It causes chest pain and can be from infections or other causes.",
            "Heart failure": "This means your heart isn't pumping blood as well as it should. It can cause shortness of breath, fatigue, and swelling.",
            "Pulmonary hypertension": "This is high blood pressure in the arteries of your lungs. It makes your heart work harder and can cause shortness of breath.",
            "Arrhythmia": "This is an irregular heartbeat - your heart might beat too fast, too slow, or irregularly. Some are harmless, others need treatment.",
            "Atrial fibrillation": "This is a type of irregular heartbeat where the upper chambers of your heart beat irregularly. It can increase stroke risk.",
            "Tachycardia": "This means your heart is beating too fast, usually over 100 beats per minute. It can be from various causes including stress, exercise, or medical conditions.",
            "Bradycardia": "This means your heart is beating too slowly, usually under 60 beats per minute. It can be normal in athletes or indicate a problem.",
            "Exercise-induced angina": "This is chest pain that occurs during physical activity due to reduced blood flow to the heart. It's a warning sign of heart disease.",
            "Hyperthyroidism": "This is when your thyroid gland produces too much hormone, speeding up your metabolism. It can cause palpitations, weight loss, and anxiety.",
            "Venous insufficiency": "This means the veins in your legs aren't working properly to return blood to your heart. It can cause leg swelling.",
            "Deep vein thrombosis (DVT)": "This is a blood clot in a deep vein, usually in your leg. It's serious because the clot can travel to your lungs.",
            "Lymphedema": "This is swelling caused by a problem with your lymphatic system, which helps drain fluid from tissues.",
            "Essential hypertension": "This is high blood pressure with no known cause. It's the most common type and usually develops over time.",
            "Postural hypotension": "This is low blood pressure that happens when you stand up quickly, causing dizziness.",
            
            # Neurological causes
            "Migraine": "This is a severe type of headache that can cause intense pain, nausea, and sensitivity to light. It can last for hours or even days.",
            "Migraine with aura": "This is a migraine that includes visual or sensory disturbances before the headache, like seeing flashing lights or zigzag lines.",
            "Tension headache": "This is the most common type of headache, usually from stress or muscle tension. It feels like a tight band around your head.",
            "Cluster headache": "This is a very painful type of headache that occurs in clusters or groups. The pain is usually around one eye.",
            "Meningitis": "This is inflammation of the membranes around your brain and spinal cord. It's serious and can be life-threatening.",
            "Brain tumor (rare)": "This is an abnormal growth in your brain. While rare, it can cause headaches and other neurological symptoms.",
            "Brain hemorrhage": "This is bleeding in the brain, which is a medical emergency. It can cause severe headaches and neurological symptoms.",
            "Stroke (rare)": "This happens when blood flow to part of your brain is cut off. It's a medical emergency that can cause permanent damage.",
            "Stroke": "This happens when blood flow to part of your brain is cut off, causing brain damage. It's a medical emergency.",
            "Transient ischemic attack (TIA)": "This is a 'mini-stroke' - a temporary blockage of blood flow to the brain. It's a warning sign of a possible future stroke.",
            "Benign paroxysmal positional vertigo (BPPV)": "This is a common cause of dizziness. It happens when tiny crystals in your inner ear get dislodged.",
            "Meniere's disease": "This is an inner ear disorder that causes episodes of vertigo, hearing loss, and ringing in the ears.",
            "Pinched nerve": "This happens when surrounding tissues press on a nerve, causing pain, numbness, or tingling.",
            "Multiple sclerosis": "This is a disease where your immune system attacks the protective covering of your nerves, causing various neurological symptoms.",
            "Peripheral neuropathy": "This is damage to nerves outside your brain and spinal cord, often causing numbness, tingling, or pain in your hands and feet.",
            "Carpal tunnel syndrome": "This is when the median nerve in your wrist is compressed, causing numbness and tingling in your hand.",
            "Vitamin B12 deficiency": "This means you don't have enough vitamin B12, which is important for nerve health. It can cause numbness and other symptoms.",
            "Epilepsy": "This is a neurological disorder that causes recurrent seizures. It can be managed with medication.",
            "Delirium": "This is a sudden change in mental function, causing confusion and disorientation. It's often from an underlying medical condition.",
            "Dementia": "This is a decline in mental ability severe enough to interfere with daily life. Alzheimer's is the most common type.",
            "Refractive errors (nearsightedness, farsightedness)": "This means your eye doesn't bend light correctly, causing blurry vision. It's usually corrected with glasses or contacts.",
            "Cataracts": "This is clouding of the lens in your eye, causing blurry vision. It's common with aging and can be treated with surgery.",
            "Glaucoma": "This is damage to your optic nerve, often from high pressure in your eye. It can lead to vision loss if untreated.",
            "Diabetic retinopathy": "This is damage to the blood vessels in your retina from diabetes. It can cause vision loss.",
            "Macular degeneration": "This is deterioration of the central part of your retina, causing loss of central vision. It's common in older adults.",
            "Retinal detachment": "This is when the retina pulls away from the back of the eye. It's a medical emergency that can cause vision loss.",
            "Age-related hearing loss": "This is gradual hearing loss that happens as you get older. It's common and can be helped with hearing aids.",
            "Acoustic neuroma (rare)": "This is a noncancerous tumor on the nerve that connects your ear to your brain. It's rare but can cause hearing loss.",
            "Guillain-Barré syndrome": "This is a rare disorder where your immune system attacks your nerves, causing weakness and sometimes paralysis.",
            "Parkinson's disease": "This is a progressive nervous system disorder that affects movement, causing tremors, stiffness, and balance problems.",
            "Essential tremor": "This is a neurological disorder that causes involuntary shaking, usually in your hands. It's different from Parkinson's.",
            "Alcohol withdrawal": "This happens when someone who drinks heavily suddenly stops, causing tremors, anxiety, and other symptoms.",
            
            # Respiratory causes
            "Pneumonia": "This is an infection in your lungs that makes it hard to breathe. It can be caused by bacteria or viruses and often comes with fever and cough.",
            "Asthma": "This is a condition where your airways become narrow and swollen, making it hard to breathe. It can be triggered by allergies, exercise, or other factors.",
            "Exercise-induced asthma": "This is asthma symptoms that occur during or after physical activity. It's common and can be managed.",
            "Chronic obstructive pulmonary disease (COPD)": "This is a long-term lung disease that makes it hard to breathe. It's usually caused by smoking and gets worse over time.",
            "Bronchitis": "This is inflammation of the tubes that carry air to your lungs. It causes a persistent cough, often with mucus. It can be acute (short-term) or chronic (long-term).",
            "Sinusitis": "This is inflammation of your sinuses, the air-filled spaces in your face. It causes congestion, facial pain, and sometimes headaches.",
            "Deviated septum": "This means the wall between your nostrils is crooked, which can cause congestion and breathing problems.",
            "Nasal polyps": "These are soft, noncancerous growths in your nose that can cause congestion and breathing problems.",
            "Post-nasal drip": "This is when mucus from your nose drips down the back of your throat, causing cough and throat irritation.",
            "Strep throat": "This is a bacterial infection of the throat caused by streptococcus bacteria. It causes severe sore throat, fever, and difficulty swallowing.",
            "Tonsillitis": "This is inflammation of the tonsils, usually from infection. It causes sore throat, difficulty swallowing, and sometimes fever.",
            
            # Gastrointestinal causes
            "Gastroenteritis": "This is inflammation of your stomach and intestines, often called 'stomach flu.' It causes nausea, vomiting, diarrhea, and stomach pain.",
            "Appendicitis": "This is when your appendix (a small organ in your lower right abdomen) becomes inflamed. It causes severe abdominal pain and usually needs surgery.",
            "Gallstones": "These are hard deposits that form in your gallbladder. They can cause abdominal pain, especially after eating fatty foods.",
            "Irritable bowel syndrome (IBS)": "This is a common disorder affecting the large intestine, causing cramping, abdominal pain, bloating, gas, diarrhea, and constipation.",
            "Food poisoning": "This happens when you eat food contaminated with bacteria or viruses. It causes nausea, vomiting, diarrhea, and stomach pain, usually within hours of eating.",
            "Constipation": "This means you're having difficulty passing stools or going less often than usual. Your stools may be hard and difficult to pass.",
            "Ulcer": "This is a sore in the lining of your stomach or small intestine. It can cause abdominal pain, especially when your stomach is empty.",
            "Peptic ulcer": "This is an ulcer in the stomach or upper part of the small intestine. It can cause abdominal pain and sometimes bleeding.",
            "Diverticulitis": "This is inflammation of small pouches in your colon. It causes abdominal pain, usually on the left side, and can be serious.",
            "Inflammatory bowel disease (IBD)": "This includes Crohn's disease and ulcerative colitis - chronic conditions that cause inflammation in your digestive tract.",
            "Lactose intolerance": "This means your body can't digest lactose, a sugar in milk. It causes bloating, gas, and diarrhea after eating dairy.",
            "Celiac disease": "This is an immune reaction to eating gluten, a protein in wheat. It damages your small intestine and causes digestive symptoms.",
            "Hiatal hernia": "This is when part of your stomach pushes up through your diaphragm, which can cause heartburn and GERD.",
            "Gastroesophageal reflux disease (GERD)": "This is when stomach acid frequently flows back into your esophagus, causing heartburn and other symptoms.",
            "Hemorrhoids": "These are swollen veins in your rectum or anus. They can cause pain, itching, and sometimes bleeding.",
            "Anal fissure": "This is a small tear in the lining of the anus. It can cause pain and bleeding during bowel movements.",
            "Colon cancer": "This is cancer that starts in the colon. Early detection is important for successful treatment.",
            "Colon polyps": "These are small clumps of cells that form on the lining of the colon. Most are harmless, but some can become cancerous.",
            "Upper GI bleeding": "This is bleeding in the upper part of your digestive tract, which can cause black, tarry stools.",
            
            # Urinary causes
            "Urinary tract infection (UTI)": "This is an infection in any part of your urinary system - kidneys, bladder, or urethra. It causes painful urination and frequent urination.",
            "Bladder infection": "This is a type of UTI specifically in your bladder. It causes painful and frequent urination.",
            "Kidney infection": "This is a serious UTI that has spread to your kidneys. It causes back pain, fever, and can be serious if untreated.",
            "Kidney stones": "These are hard deposits that form in your kidneys. They can cause severe pain when they pass through your urinary tract.",
            "Bladder stones": "These are hard deposits that form in your bladder. They can cause pain and frequent urination.",
            "Overactive bladder": "This means your bladder contracts too often, causing frequent and urgent urination.",
            "Prostate problems": "The prostate is a gland in men. When it's enlarged or infected, it can cause urinary problems.",
            "Interstitial cystitis": "This is a chronic condition causing bladder pressure and pain, along with frequent, painful urination.",
            "Sexually transmitted infection (STI)": "These are infections passed through sexual contact. Some can cause painful urination.",
            "Bladder cancer": "This is cancer that starts in the bladder. Early symptoms can include blood in urine and frequent urination.",
            
            # Skin causes
            "Allergic reaction": "This happens when your immune system overreacts to something harmless (like pollen, food, or medication). It can cause rashes, swelling, or more serious symptoms.",
            "Eczema": "This is a skin condition that causes dry, itchy, and inflamed skin. It's common and can be managed with proper skin care and medication.",
            "Psoriasis": "This is a skin condition that causes red, scaly patches. It's an autoimmune condition that can be managed with treatment.",
            "Contact dermatitis": "This is a rash caused by touching something that irritates your skin or causes an allergic reaction.",
            "Fungal infection": "This is an infection caused by fungi, like athlete's foot or ringworm. It can cause rashes and itching.",
            "Parasitic infection": "This is when parasites (like scabies or lice) infest your skin, causing itching and rashes.",
            "Viral infection": "This is an illness caused by a virus. Many viral infections can cause skin rashes as a symptom.",
            "Bacterial infection": "This is when harmful bacteria infect your skin, causing rashes, redness, or other symptoms.",
            
            # Musculoskeletal causes
            "Arthritis": "This is inflammation of your joints, causing pain, stiffness, and swelling. There are many types, with osteoarthritis and rheumatoid arthritis being most common.",
            "Osteoarthritis": "This is the most common type of arthritis, caused by wear and tear on your joints over time.",
            "Rheumatoid arthritis": "This is an autoimmune disease that causes inflammation in your joints, leading to pain and deformity.",
            "Gout": "This is a type of arthritis caused by uric acid crystals in your joints, causing sudden, severe pain, often in your big toe.",
            "Injury": "This is damage to your body from an accident or trauma. It can cause pain, swelling, and limited movement.",
            "Knee injury": "This is damage to the knee from trauma or overuse. It can cause pain, swelling, and difficulty moving.",
            "Torn ligament": "This is when a ligament (tissue connecting bones) is torn, usually from injury. It can cause pain and instability.",
            "Overuse": "This is pain from repetitive movements or overworking a muscle or joint. Rest usually helps.",
            "Lupus": "This is an autoimmune disease that can affect many parts of your body, including joints, skin, and organs.",
            "Fibromyalgia": "This is a condition causing widespread muscle pain, fatigue, and tender points throughout your body.",
            "Sciatica": "This is pain along the sciatic nerve, which runs from your lower back down your leg. It's often from a herniated disc.",
            "Herniated disc": "This is when the soft cushion between your vertebrae bulges or ruptures, pressing on nerves and causing pain.",
            "Spinal stenosis": "This is narrowing of the spaces in your spine, which can press on nerves and cause pain.",
            "Osteoporosis": "This is a condition where your bones become weak and brittle, increasing fracture risk.",
            "Whiplash": "This is a neck injury from a sudden back-and-forth movement, often from car accidents.",
            "Muscle strain": "This is when a muscle is stretched or torn, usually from overuse or injury. It causes pain and limited movement.",
            "Poor posture": "This is when you sit or stand in positions that put stress on your muscles and joints, causing pain over time.",
            
            # General causes
            "Viral infection (flu, cold)": "These are common illnesses caused by viruses. They usually cause fever, body aches, and fatigue, and most people recover with rest and fluids.",
            "Bacterial infection": "This is when harmful bacteria get into your body and cause illness. These often need antibiotics to treat and can cause fever and other symptoms.",
            "Urinary tract infection": "This is an infection in any part of your urinary system. It can cause fever, pain, and frequent urination.",
            "Respiratory infection": "This is an infection in your respiratory system - your nose, throat, or lungs. It can cause cough, congestion, and fever.",
            "Autoimmune condition": "This is when your immune system mistakenly attacks your own body. There are many types, like rheumatoid arthritis and lupus.",
            "Medication reaction": "This is when a medication causes an unwanted side effect or allergic reaction.",
            "Medication side effects": "These are unwanted effects from medications. They can range from mild to serious.",
            "Cancer (rare)": "This is when cells in your body grow uncontrollably. While it can cause various symptoms, most symptoms are from other causes.",
            "Sleep apnea": "This is when you stop breathing repeatedly during sleep. It causes fatigue and can lead to serious health problems.",
            "Sleep deprivation": "This means you're not getting enough sleep. It can cause fatigue, difficulty concentrating, and other symptoms.",
            "Chronic fatigue syndrome": "This is a complex disorder causing extreme fatigue that doesn't improve with rest and can't be explained by an underlying medical condition.",
            "Diabetes": "This is when your body can't properly use or produce insulin, causing high blood sugar. It can cause many symptoms including excessive thirst, frequent urination, and fatigue.",
            "Diabetes insipidus": "This is a rare condition causing excessive thirst and urination, but it's different from diabetes mellitus.",
            "Cushing's syndrome": "This is when your body has too much cortisol, a stress hormone. It can cause weight gain and other symptoms.",
            "Hyperthyroidism": "This is when your thyroid produces too much hormone, speeding up your metabolism.",
            "Hypothyroidism": "This is when your thyroid doesn't produce enough hormone, slowing down your metabolism.",
            "Anemia": "This means you don't have enough red blood cells or hemoglobin. It can cause fatigue, weakness, and shortness of breath.",
            "Dehydration": "This means your body doesn't have enough water. It can cause dizziness, fatigue, and other symptoms.",
            "Menopause": "This is when a woman's periods stop permanently, usually around age 50. It can cause night sweats, mood changes, and other symptoms.",
            "Perimenopause": "This is the time before menopause when hormone levels start changing. It can cause irregular periods and other symptoms.",
            "Polycystic ovary syndrome (PCOS)": "This is a hormonal disorder in women that can cause irregular periods, weight gain, and other symptoms.",
            "Endometriosis": "This is when tissue similar to the lining of your uterus grows outside it, causing pelvic pain and other symptoms.",
            "Ovarian cysts": "These are fluid-filled sacs in or on your ovaries. Most are harmless, but some can cause pain.",
            "Pelvic inflammatory disease (PID)": "This is an infection of the female reproductive organs, often from sexually transmitted bacteria.",
            "Vasovagal syncope": "This is fainting caused by a sudden drop in heart rate and blood pressure, often from stress or pain.",
            "Restless leg syndrome": "This is a condition causing an uncontrollable urge to move your legs, often at night, disrupting sleep.",
            "Pregnancy": "This can cause various symptoms including nausea, frequent urination, and other changes in the body.",
            
            # Mental health causes
            "Generalized anxiety disorder": "This is excessive worry about everyday things that's hard to control and interferes with daily life.",
            "Panic disorder": "This causes sudden, intense episodes of fear (panic attacks) with physical symptoms like rapid heartbeat and shortness of breath.",
            "Panic attack": "This is a sudden episode of intense fear with physical symptoms like rapid heartbeat, sweating, and shortness of breath.",
            "Major depressive disorder": "This is persistent sadness and loss of interest that affects how you feel, think, and handle daily activities.",
            "Depression": "This is more than just feeling sad - it's a persistent feeling of sadness, hopelessness, or loss of interest that affects daily life.",
            "Bipolar disorder": "This causes extreme mood swings from emotional highs (mania) to lows (depression).",
            "Stress": "This is your body's response to pressure or demands. While normal, excessive stress can affect your health.",
            "Anxiety": "This is feeling worried, nervous, or uneasy, often about something with an uncertain outcome. While normal in some situations, excessive anxiety can interfere with daily life.",
            "Substance use": "Using drugs or alcohol can cause various physical and mental health symptoms.",
            "Poor sleep habits": "This includes irregular sleep schedules, using screens before bed, or other habits that interfere with good sleep.",
            
            # Other causes
            "Pleurisy": "This is inflammation of the pleura, the membrane around your lungs. It causes sharp chest pain when breathing.",
            "Inner ear problems": "These can cause dizziness, vertigo, and balance problems. They're often related to the vestibular system.",
            "Electrolyte imbalance": "This means the levels of minerals in your blood are off. It can cause various symptoms including weakness and confusion.",
            "Low blood sugar": "This means your blood sugar is too low. It can cause dizziness, confusion, and other symptoms.",
            "Hormonal imbalance": "This means your hormones are not at normal levels. It can cause various symptoms depending on which hormones are affected.",
            "Hormonal changes": "These are changes in hormone levels, which can occur naturally or from medical conditions. They can cause various symptoms.",
            "Lifestyle factors (diet, exercise)": "Your diet and exercise habits can significantly affect your health and cause various symptoms.",
            "Blood loss": "This is when you lose blood, either from injury or internal bleeding. It can cause weakness, dizziness, and low blood pressure.",
            "Endocrine disorders": "These are conditions affecting your hormone-producing glands. They can cause various symptoms.",
            "Motion sickness": "This is nausea and dizziness caused by movement, like in a car or boat.",
            "Chronic infection": "This is an infection that lasts a long time or keeps coming back. It can cause ongoing symptoms.",
            "Digestive disorders": "These are conditions affecting your digestive system. They can cause various symptoms like pain, bloating, and changes in bowel habits.",
            "Chronic illness": "This is a long-term medical condition that can cause ongoing symptoms and affect your quality of life.",
            "Life events": "Major life changes or stressful events can affect your physical and mental health, causing various symptoms.",
        }
        
        # Return detailed explanation if available, otherwise provide a generic one
        explanation = cause_explanations_db.get(cause, None)
        if explanation:
            return explanation
        
        # Try to find a partial match for similar causes
        cause_lower = cause.lower()
        for key, value in cause_explanations_db.items():
            if cause_lower in key.lower() or key.lower() in cause_lower:
                return value
        
        # Default explanation
        return f"{cause} is a possible cause of your symptoms. A healthcare provider can help determine if this is the cause and recommend appropriate treatment."
    
    def _generate_explanations(self, urgency: str, cause: str, risks: List[str], all_possible_causes: List[str] = None) -> Dict:
        """Generate detailed explanations for the diagnosis, with focus on causes"""
        # Generate detailed cause explanations for all possible causes
        cause_explanations = []
        
        # Get explanations for all possible causes
        causes_to_explain = all_possible_causes if all_possible_causes else [cause]
        
        # Use a set to avoid duplicate explanations
        seen_causes = set()
        for possible_cause in causes_to_explain:
            if possible_cause not in seen_causes:
                seen_causes.add(possible_cause)
                explanation = self._get_cause_explanation(possible_cause)
                cause_explanations.append({
                    "term": possible_cause,
                    "explanation": explanation
                })
        
        return {
            "symptom_explanations": [],
            "cause_explanations": cause_explanations,
            "risk_explanations": [{"term": risk, "explanation": f"{risk}."} for risk in risks]
        }
    
    def save_models(self):
        """Save trained models to disk"""
        model_path = self.model_dir / "diagnosis_models"
        model_path.mkdir(exist_ok=True)
        
        with open(model_path / "urgency_model.pkl", "wb") as f:
            pickle.dump(self.urgency_model, f)
        with open(model_path / "cause_model.pkl", "wb") as f:
            pickle.dump(self.cause_model, f)
        with open(model_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(model_path / "urgency_encoder.pkl", "wb") as f:
            pickle.dump(self.urgency_encoder, f)
        with open(model_path / "cause_encoder.pkl", "wb") as f:
            pickle.dump(self.cause_encoder, f)
        
        print(f"Models saved to {model_path}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        model_path = self.model_dir / "diagnosis_models"
        try:
            with open(model_path / "urgency_model.pkl", "rb") as f:
                self.urgency_model = pickle.load(f)
            with open(model_path / "cause_model.pkl", "rb") as f:
                self.cause_model = pickle.load(f)
            with open(model_path / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
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
_enhanced_ml_model_instance = None


def get_enhanced_ml_model() -> EnhancedMLDiagnosisModel:
    """Get or create global enhanced ML model instance"""
    global _enhanced_ml_model_instance
    if _enhanced_ml_model_instance is None:
        _enhanced_ml_model_instance = EnhancedMLDiagnosisModel()
        _enhanced_ml_model_instance.load_models()
    return _enhanced_ml_model_instance

