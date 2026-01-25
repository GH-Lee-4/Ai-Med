"""
Symptoms Diagnosis Module with AI Agent
Analyzes symptoms and determines urgency, causes, and possible risks
"""

import re
from typing import Dict, List, Tuple
from enum import Enum


class UrgencyLevel(Enum):
    """Urgency levels for medical conditions"""
    CRITICAL = "Critical - Seek immediate emergency care"
    HIGH = "High - Consult a doctor within 24 hours"
    MODERATE = "Moderate - Schedule appointment within a few days"
    LOW = "Low - Monitor and consult if symptoms persist"
    ROUTINE = "Routine - Self-care may be sufficient"


class SymptomsDiagnosisAgent:
    """
    AI Agent for symptoms diagnosis
    Analyzes symptoms and provides urgency assessment, causes, and risks
    """
    
    def __init__(self):
        """Initialize the diagnosis agent with medical knowledge base"""
        self.symptom_patterns = self._initialize_symptom_patterns()
        self.urgency_rules = self._initialize_urgency_rules()
        self.cause_database = self._initialize_cause_database()
        self.risk_factors = self._initialize_risk_factors()
        self.explanations_database = self._initialize_explanations_database()
    
    def _initialize_symptom_patterns(self) -> Dict[str, List[str]]:
        """Initialize symptom pattern matching database"""
        return {
            # Cardiovascular symptoms
            "chest_pain": ["chest pain", "chest discomfort", "tightness", "pressure", "heart pain"],
            "shortness_breath": ["shortness of breath", "difficulty breathing", "breathlessness", "can't breathe", "out of breath"],
            "palpitations": ["palpitations", "irregular heartbeat", "racing heart", "heart racing", "heart pounding", "skipped beat"],
            "swelling_legs": ["swollen legs", "swelling in legs", "leg swelling", "edema", "fluid retention", "puffy legs"],
            "high_blood_pressure": ["high blood pressure", "hypertension", "elevated blood pressure"],
            "low_blood_pressure": ["low blood pressure", "hypotension", "dizzy when standing"],
            
            # Neurological symptoms
            "severe_headache": ["severe headache", "thunderclap headache", "worst headache", "intense headache"],
            "dizziness": ["dizziness", "lightheaded", "vertigo", "feeling faint", "unsteady"],
            "numbness": ["numbness", "numb", "loss of sensation", "tingling", "pins and needles"],
            "seizure": ["seizure", "convulsion", "fit", "epilepsy"],
            "confusion": ["confusion", "disorientation", "memory loss", "can't think clearly"],
            "vision_problems": ["blurred vision", "double vision", "vision loss", "eye pain", "seeing spots", "flashing lights"],
            "hearing_loss": ["hearing loss", "ringing in ears", "tinnitus", "ear pain", "earache"],
            "speech_problems": ["slurred speech", "difficulty speaking", "trouble talking"],
            "weakness": ["weakness", "muscle weakness", "can't move", "paralysis"],
            
            # Respiratory symptoms
            "cough": ["cough", "coughing", "persistent cough", "dry cough", "wet cough"],
            "wheezing": ["wheezing", "whistling sound", "breathing sounds"],
            "congestion": ["congestion", "stuffy nose", "nasal congestion", "runny nose"],
            "sore_throat": ["sore throat", "throat pain", "difficulty swallowing", "painful swallowing"],
            
            # Gastrointestinal symptoms
            "abdominal_pain": ["abdominal pain", "stomach pain", "belly ache", "tummy ache", "cramps"],
            "nausea_vomiting": ["nausea", "vomiting", "throwing up", "feeling sick"],
            "diarrhea": ["diarrhea", "loose stools", "watery stools", "frequent bowel movements"],
            "constipation": ["constipation", "can't go", "hard stools", "infrequent bowel movements"],
            "bloating": ["bloating", "gas", "feeling full", "abdominal distension"],
            "heartburn": ["heartburn", "acid reflux", "indigestion", "burning sensation"],
            "blood_stool": ["blood in stool", "rectal bleeding", "bloody stool", "black stool"],
            
            # Urinary symptoms
            "frequent_urination": ["frequent urination", "urinating often", "need to pee often"],
            "painful_urination": ["painful urination", "burning when urinating", "urination pain"],
            "blood_urine": ["blood in urine", "bloody urine", "hematuria"],
            "incontinence": ["incontinence", "can't control urine", "leaking urine"],
            
            # Skin symptoms
            "rash": ["rash", "skin irritation", "hives", "skin rash", "redness"],
            "itching": ["itching", "itchy", "pruritus", "skin itching"],
            "wound": ["wound", "cut", "laceration", "open sore", "ulcer"],
            "bruising": ["bruising", "bruises", "easy bruising", "unexplained bruises"],
            "swelling": ["swelling", "swollen", "edema", "puffiness"],
            "discoloration": ["skin discoloration", "yellow skin", "jaundice", "pale skin"],
            
            # Musculoskeletal symptoms
            "joint_pain": ["joint pain", "arthritis", "stiffness", "joint stiffness", "achy joints"],
            "back_pain": ["back pain", "lower back pain", "upper back pain", "spine pain"],
            "neck_pain": ["neck pain", "stiff neck", "cervical pain"],
            "muscle_pain": ["muscle pain", "muscle ache", "myalgia", "sore muscles"],
            "limb_pain": ["arm pain", "leg pain", "limb pain", "extremity pain"],
            
            # General symptoms
            "fever": ["fever", "high temperature", "chills", "feeling hot", "sweating"],
            "fatigue": ["fatigue", "tiredness", "exhaustion", "feeling tired", "low energy"],
            "weight_loss": ["weight loss", "losing weight", "unintended weight loss"],
            "weight_gain": ["weight gain", "gaining weight", "unexplained weight gain"],
            "appetite_loss": ["loss of appetite", "not hungry", "can't eat", "no appetite"],
            "excessive_thirst": ["excessive thirst", "very thirsty", "drinking a lot"],
            "night_sweats": ["night sweats", "sweating at night", "waking up sweaty"],
            
            # Mental health symptoms
            "anxiety": ["anxiety", "feeling anxious", "panic", "worry", "nervous"],
            "depression": ["depression", "feeling sad", "hopeless", "down", "depressed"],
            "insomnia": ["insomnia", "can't sleep", "sleep problems", "trouble sleeping"],
            "mood_changes": ["mood changes", "mood swings", "irritability", "moody"],
            
            # Reproductive symptoms (gender-neutral terms)
            "pelvic_pain": ["pelvic pain", "lower abdominal pain", "groin pain"],
            "menstrual_irregularities": ["irregular periods", "missed period", "heavy bleeding", "menstrual problems"],
            "erectile_dysfunction": ["erectile dysfunction", "impotence", "trouble with erection"],
            
            # Other important symptoms
            "bleeding": ["bleeding", "uncontrolled bleeding", "excessive bleeding", "hemorrhage"],
            "fainting": ["fainting", "passing out", "syncope", "loss of consciousness"],
            "tremor": ["tremor", "shaking", "trembling", "uncontrollable shaking"],
            "balance_problems": ["balance problems", "trouble balancing", "unsteady gait", "falling"],
        }
    
    def _initialize_urgency_rules(self) -> Dict[str, UrgencyLevel]:
        """Initialize rules for determining urgency based on symptoms"""
        return {
            # Critical symptoms
            "chest_pain": UrgencyLevel.CRITICAL,
            "shortness_breath": UrgencyLevel.CRITICAL,
            "seizure": UrgencyLevel.CRITICAL,
            "bleeding": UrgencyLevel.CRITICAL,
            "fainting": UrgencyLevel.CRITICAL,
            "blood_stool": UrgencyLevel.CRITICAL,
            "blood_urine": UrgencyLevel.CRITICAL,
            "confusion": UrgencyLevel.CRITICAL,
            "speech_problems": UrgencyLevel.CRITICAL,
            "weakness": UrgencyLevel.CRITICAL,
            
            # High urgency symptoms
            "severe_headache": UrgencyLevel.HIGH,
            "vision_problems": UrgencyLevel.HIGH,
            "palpitations": UrgencyLevel.HIGH,
            "high_blood_pressure": UrgencyLevel.HIGH,
            "swelling_legs": UrgencyLevel.HIGH,
            "numbness": UrgencyLevel.HIGH,
            "balance_problems": UrgencyLevel.HIGH,
            "tremor": UrgencyLevel.HIGH,
            
            # Moderate urgency symptoms
            "fever": UrgencyLevel.MODERATE,
            "abdominal_pain": UrgencyLevel.MODERATE,
            "dizziness": UrgencyLevel.MODERATE,
            "nausea_vomiting": UrgencyLevel.MODERATE,
            "diarrhea": UrgencyLevel.MODERATE,
            "constipation": UrgencyLevel.MODERATE,
            "painful_urination": UrgencyLevel.MODERATE,
            "frequent_urination": UrgencyLevel.MODERATE,
            "wound": UrgencyLevel.MODERATE,
            "hearing_loss": UrgencyLevel.MODERATE,
            "sore_throat": UrgencyLevel.MODERATE,
            "pelvic_pain": UrgencyLevel.MODERATE,
            "menstrual_irregularities": UrgencyLevel.MODERATE,
            "low_blood_pressure": UrgencyLevel.MODERATE,
            "discoloration": UrgencyLevel.MODERATE,
            "weight_loss": UrgencyLevel.MODERATE,
            "excessive_thirst": UrgencyLevel.MODERATE,
            "night_sweats": UrgencyLevel.MODERATE,
            
            # Low urgency symptoms
            "rash": UrgencyLevel.LOW,
            "fatigue": UrgencyLevel.LOW,
            "cough": UrgencyLevel.LOW,
            "joint_pain": UrgencyLevel.LOW,
            "back_pain": UrgencyLevel.LOW,
            "neck_pain": UrgencyLevel.LOW,
            "muscle_pain": UrgencyLevel.LOW,
            "limb_pain": UrgencyLevel.LOW,
            "itching": UrgencyLevel.LOW,
            "bruising": UrgencyLevel.LOW,
            "swelling": UrgencyLevel.LOW,
            "bloating": UrgencyLevel.LOW,
            "heartburn": UrgencyLevel.LOW,
            "congestion": UrgencyLevel.LOW,
            "wheezing": UrgencyLevel.LOW,
            "incontinence": UrgencyLevel.LOW,
            "weight_gain": UrgencyLevel.LOW,
            "appetite_loss": UrgencyLevel.LOW,
            "erectile_dysfunction": UrgencyLevel.LOW,
            
            # Routine symptoms
            "anxiety": UrgencyLevel.ROUTINE,
            "depression": UrgencyLevel.ROUTINE,
            "insomnia": UrgencyLevel.ROUTINE,
            "mood_changes": UrgencyLevel.ROUTINE,
        }
    
    def _initialize_cause_database(self) -> Dict[str, List[str]]:
        """Initialize database of possible causes for symptoms"""
        return {
            # Cardiovascular
            "chest_pain": [
                "Heart attack or angina",
                "Pulmonary embolism",
                "Pneumonia",
                "Gastroesophageal reflux disease (GERD)",
                "Anxiety or panic attack",
                "Muscle strain",
                "Costochondritis",
                "Pericarditis"
            ],
            "shortness_breath": [
                "Asthma",
                "Chronic obstructive pulmonary disease (COPD)",
                "Pneumonia",
                "Pulmonary embolism",
                "Heart failure",
                "Anxiety",
                "Anemia",
                "Pulmonary hypertension"
            ],
            "palpitations": [
                "Anxiety or stress",
                "Caffeine or stimulant use",
                "Arrhythmia",
                "Hyperthyroidism",
                "Anemia",
                "Heart disease",
                "Medication side effects"
            ],
            "swelling_legs": [
                "Heart failure",
                "Kidney disease",
                "Liver disease",
                "Venous insufficiency",
                "Deep vein thrombosis (DVT)",
                "Lymphedema",
                "Medication side effects"
            ],
            "high_blood_pressure": [
                "Essential hypertension",
                "Kidney disease",
                "Thyroid problems",
                "Sleep apnea",
                "Medication side effects",
                "Lifestyle factors (diet, exercise)"
            ],
            "low_blood_pressure": [
                "Dehydration",
                "Blood loss",
                "Heart problems",
                "Endocrine disorders",
                "Medication side effects",
                "Postural hypotension"
            ],
            
            # Neurological
            "severe_headache": [
                "Migraine",
                "Tension headache",
                "Cluster headache",
                "Meningitis",
                "Brain tumor (rare)",
                "Stroke (rare)",
                "Sinusitis",
                "Medication overuse"
            ],
            "dizziness": [
                "Inner ear problems",
                "Low blood pressure",
                "Dehydration",
                "Anemia",
                "Medication side effects",
                "Anxiety",
                "Benign paroxysmal positional vertigo (BPPV)",
                "Meniere's disease"
            ],
            "numbness": [
                "Pinched nerve",
                "Diabetes",
                "Multiple sclerosis",
                "Stroke",
                "Peripheral neuropathy",
                "Carpal tunnel syndrome",
                "Vitamin B12 deficiency"
            ],
            "seizure": [
                "Epilepsy",
                "Head injury",
                "Brain tumor",
                "Stroke",
                "Infection (meningitis, encephalitis)",
                "Drug withdrawal",
                "Metabolic disorders"
            ],
            "confusion": [
                "Dehydration",
                "Infection",
                "Medication side effects",
                "Stroke",
                "Dementia",
                "Delirium",
                "Low blood sugar",
                "Electrolyte imbalance"
            ],
            "vision_problems": [
                "Refractive errors (nearsightedness, farsightedness)",
                "Cataracts",
                "Glaucoma",
                "Diabetic retinopathy",
                "Macular degeneration",
                "Migraine",
                "Stroke",
                "Eye infection"
            ],
            "hearing_loss": [
                "Age-related hearing loss",
                "Noise exposure",
                "Ear infection",
                "Earwax buildup",
                "Meniere's disease",
                "Medication side effects",
                "Acoustic neuroma (rare)"
            ],
            "speech_problems": [
                "Stroke",
                "Transient ischemic attack (TIA)",
                "Brain injury",
                "Neurological disorders",
                "Medication side effects",
                "Alcohol intoxication"
            ],
            "weakness": [
                "Stroke",
                "Multiple sclerosis",
                "Guillain-Barré syndrome",
                "Muscle disease",
                "Nerve damage",
                "Electrolyte imbalance",
                "Chronic fatigue"
            ],
            
            # Respiratory
            "cough": [
                "Common cold",
                "Flu",
                "Bronchitis",
                "Pneumonia",
                "Asthma",
                "Allergies",
                "Post-nasal drip",
                "GERD"
            ],
            "wheezing": [
                "Asthma",
                "COPD",
                "Bronchitis",
                "Allergic reaction",
                "Heart failure",
                "Anxiety"
            ],
            "congestion": [
                "Common cold",
                "Allergies",
                "Sinusitis",
                "Deviated septum",
                "Nasal polyps"
            ],
            "sore_throat": [
                "Viral infection (cold, flu)",
                "Bacterial infection (strep throat)",
                "Allergies",
                "GERD",
                "Dry air",
                "Smoking"
            ],
            
            # Gastrointestinal
            "abdominal_pain": [
                "Gastroenteritis",
                "Appendicitis",
                "Gallstones",
                "Irritable bowel syndrome (IBS)",
                "Food poisoning",
                "Constipation",
                "Ulcer",
                "Diverticulitis"
            ],
            "nausea_vomiting": [
                "Gastroenteritis",
                "Food poisoning",
                "Pregnancy",
                "Migraine",
                "Medication side effects",
                "Motion sickness",
                "GERD",
                "Appendicitis"
            ],
            "diarrhea": [
                "Viral infection",
                "Bacterial infection",
                "Food poisoning",
                "Irritable bowel syndrome (IBS)",
                "Inflammatory bowel disease (IBD)",
                "Medication side effects",
                "Lactose intolerance"
            ],
            "constipation": [
                "Inadequate fiber intake",
                "Dehydration",
                "Lack of exercise",
                "Medication side effects",
                "Irritable bowel syndrome (IBS)",
                "Hypothyroidism",
                "Colon obstruction"
            ],
            "bloating": [
                "Gas",
                "Irritable bowel syndrome (IBS)",
                "Food intolerance",
                "Constipation",
                "Overeating",
                "Celiac disease"
            ],
            "heartburn": [
                "Gastroesophageal reflux disease (GERD)",
                "Hiatal hernia",
                "Pregnancy",
                "Certain foods",
                "Obesity",
                "Smoking"
            ],
            "blood_stool": [
                "Hemorrhoids",
                "Anal fissure",
                "Inflammatory bowel disease (IBD)",
                "Colon cancer",
                "Diverticulitis",
                "Peptic ulcer",
                "Colon polyps"
            ],
            
            # Urinary
            "frequent_urination": [
                "Urinary tract infection (UTI)",
                "Diabetes",
                "Overactive bladder",
                "Prostate problems",
                "Pregnancy",
                "Medication side effects",
                "Interstitial cystitis"
            ],
            "painful_urination": [
                "Urinary tract infection (UTI)",
                "Bladder infection",
                "Kidney infection",
                "Sexually transmitted infection (STI)",
                "Bladder stones",
                "Interstitial cystitis"
            ],
            "blood_urine": [
                "Urinary tract infection (UTI)",
                "Kidney stones",
                "Bladder cancer",
                "Kidney disease",
                "Kidney infection",
                "Trauma",
                "Medication side effects"
            ],
            "incontinence": [
                "Overactive bladder",
                "Weak pelvic muscles",
                "Neurological disorders",
                "Prostate problems",
                "Medication side effects",
                "Urinary tract infection (UTI)"
            ],
            
            # Skin
            "rash": [
                "Allergic reaction",
                "Eczema",
                "Psoriasis",
                "Contact dermatitis",
                "Viral infection",
                "Bacterial infection",
                "Fungal infection",
                "Autoimmune condition"
            ],
            "itching": [
                "Dry skin",
                "Eczema",
                "Allergic reaction",
                "Liver disease",
                "Kidney disease",
                "Diabetes",
                "Parasitic infection"
            ],
            "wound": [
                "Injury",
                "Surgical incision",
                "Pressure ulcer",
                "Diabetic ulcer",
                "Infection"
            ],
            "bruising": [
                "Injury",
                "Medication side effects (blood thinners)",
                "Vitamin deficiency",
                "Blood disorders",
                "Liver disease",
                "Aging"
            ],
            "swelling": [
                "Injury",
                "Infection",
                "Allergic reaction",
                "Heart failure",
                "Kidney disease",
                "Liver disease",
                "Lymphedema"
            ],
            "discoloration": [
                "Jaundice (liver problems)",
                "Anemia",
                "Circulation problems",
                "Skin conditions",
                "Medication side effects"
            ],
            
            # Musculoskeletal
            "joint_pain": [
                "Arthritis",
                "Osteoarthritis",
                "Rheumatoid arthritis",
                "Gout",
                "Injury",
                "Overuse",
                "Lupus",
                "Fibromyalgia"
            ],
            "back_pain": [
                "Muscle strain",
                "Herniated disc",
                "Osteoarthritis",
                "Poor posture",
                "Kidney infection",
                "Sciatica",
                "Spinal stenosis",
                "Osteoporosis"
            ],
            "neck_pain": [
                "Muscle strain",
                "Poor posture",
                "Herniated disc",
                "Osteoarthritis",
                "Whiplash",
                "Stress"
            ],
            "muscle_pain": [
                "Overuse",
                "Injury",
                "Fibromyalgia",
                "Infections",
                "Autoimmune conditions",
                "Medication side effects"
            ],
            "limb_pain": [
                "Injury",
                "Arthritis",
                "Nerve compression",
                "Circulation problems",
                "Infection",
                "Overuse"
            ],
            
            # General
            "fever": [
                "Viral infection (flu, cold)",
                "Bacterial infection",
                "Urinary tract infection",
                "Respiratory infection",
                "Autoimmune condition",
                "Medication reaction",
                "Cancer (rare)"
            ],
            "fatigue": [
                "Sleep deprivation",
                "Anemia",
                "Thyroid problems",
                "Depression",
                "Chronic fatigue syndrome",
                "Diabetes",
                "Sleep apnea",
                "Heart disease"
            ],
            "weight_loss": [
                "Diet changes",
                "Hyperthyroidism",
                "Diabetes",
                "Cancer",
                "Depression",
                "Digestive disorders",
                "Chronic infection"
            ],
            "weight_gain": [
                "Diet and lifestyle",
                "Hypothyroidism",
                "Medication side effects",
                "Hormonal changes",
                "Depression",
                "Cushing's syndrome"
            ],
            "appetite_loss": [
                "Infection",
                "Depression",
                "Medication side effects",
                "Digestive disorders",
                "Cancer",
                "Chronic illness"
            ],
            "excessive_thirst": [
                "Diabetes",
                "Dehydration",
                "Diabetes insipidus",
                "Medication side effects",
                "Kidney disease"
            ],
            "night_sweats": [
                "Menopause",
                "Infection",
                "Medication side effects",
                "Hyperthyroidism",
                "Cancer (rare)",
                "Anxiety"
            ],
            
            # Mental health
            "anxiety": [
                "Generalized anxiety disorder",
                "Panic disorder",
                "Stress",
                "Medical conditions",
                "Medication side effects",
                "Substance use"
            ],
            "depression": [
                "Major depressive disorder",
                "Bipolar disorder",
                "Medical conditions",
                "Medication side effects",
                "Life events",
                "Hormonal changes"
            ],
            "insomnia": [
                "Stress",
                "Anxiety",
                "Depression",
                "Medication side effects",
                "Sleep apnea",
                "Restless leg syndrome",
                "Poor sleep habits"
            ],
            "mood_changes": [
                "Hormonal changes",
                "Depression",
                "Bipolar disorder",
                "Medication side effects",
                "Stress",
                "Thyroid problems"
            ],
            
            # Reproductive
            "pelvic_pain": [
                "Menstrual cramps",
                "Endometriosis",
                "Ovarian cysts",
                "Pelvic inflammatory disease (PID)",
                "Urinary tract infection (UTI)",
                "Irritable bowel syndrome (IBS)"
            ],
            "menstrual_irregularities": [
                "Hormonal imbalance",
                "Polycystic ovary syndrome (PCOS)",
                "Thyroid problems",
                "Stress",
                "Weight changes",
                "Perimenopause"
            ],
            "erectile_dysfunction": [
                "Vascular problems",
                "Diabetes",
                "Medication side effects",
                "Psychological factors",
                "Hormonal imbalance",
                "Neurological conditions"
            ],
            
            # Other
            "bleeding": [
                "Injury",
                "Medication side effects (blood thinners)",
                "Blood disorders",
                "Liver disease",
                "Cancer",
                "Trauma"
            ],
            "fainting": [
                "Low blood pressure",
                "Dehydration",
                "Heart problems",
                "Vasovagal syncope",
                "Neurological conditions",
                "Medication side effects"
            ],
            "tremor": [
                "Essential tremor",
                "Parkinson's disease",
                "Hyperthyroidism",
                "Medication side effects",
                "Anxiety",
                "Alcohol withdrawal"
            ],
            "balance_problems": [
                "Inner ear problems",
                "Neurological conditions",
                "Medication side effects",
                "Vision problems",
                "Muscle weakness",
                "Low blood pressure"
            ],
        }
    
    def _initialize_risk_factors(self) -> Dict[str, List[str]]:
        """Initialize risk factors associated with symptoms"""
        return {
            # Cardiovascular
            "chest_pain": [
                "Risk of heart attack or cardiac event",
                "Potential for life-threatening condition",
                "May indicate serious cardiovascular issue"
            ],
            "shortness_breath": [
                "Risk of respiratory failure",
                "Potential for severe asthma attack",
                "May indicate serious lung or heart condition"
            ],
            "palpitations": [
                "Risk of arrhythmia complications",
                "Potential for heart failure",
                "May indicate underlying heart condition"
            ],
            "swelling_legs": [
                "Risk of deep vein thrombosis (DVT)",
                "Potential for heart or kidney failure",
                "May indicate serious systemic condition"
            ],
            "high_blood_pressure": [
                "Risk of stroke",
                "Potential for heart attack",
                "May lead to kidney damage"
            ],
            "low_blood_pressure": [
                "Risk of falls and injury",
                "Potential for fainting",
                "May indicate serious underlying condition"
            ],
            
            # Neurological
            "severe_headache": [
                "Risk of stroke or brain hemorrhage",
                "Potential for meningitis",
                "May indicate serious neurological condition"
            ],
            "dizziness": [
                "Risk of falls and injury",
                "Potential for fainting",
                "May indicate cardiovascular or neurological issue"
            ],
            "numbness": [
                "Risk of permanent nerve damage",
                "Potential for stroke",
                "May indicate serious neurological condition"
            ],
            "seizure": [
                "Risk of injury during seizure",
                "Potential for status epilepticus",
                "May indicate serious brain condition"
            ],
            "confusion": [
                "Risk of falls and accidents",
                "Potential for serious underlying condition",
                "May indicate stroke or infection"
            ],
            "vision_problems": [
                "Risk of permanent vision loss",
                "Potential for stroke",
                "May indicate serious eye or brain condition"
            ],
            "hearing_loss": [
                "Risk of permanent hearing loss",
                "Potential for underlying serious condition",
                "May affect quality of life"
            ],
            "speech_problems": [
                "Risk of stroke",
                "Potential for serious neurological condition",
                "May indicate brain injury"
            ],
            "weakness": [
                "Risk of falls and injury",
                "Potential for stroke",
                "May indicate serious neurological condition"
            ],
            
            # Respiratory
            "cough": [
                "Risk of respiratory complications",
                "Potential for pneumonia",
                "May indicate serious lung condition"
            ],
            "wheezing": [
                "Risk of severe asthma attack",
                "Potential for respiratory failure",
                "May indicate serious lung condition"
            ],
            "congestion": [
                "Risk of sinus infection",
                "Potential for ear infection",
                "May affect sleep quality"
            ],
            "sore_throat": [
                "Risk of complications from strep throat",
                "Potential for abscess formation",
                "May indicate serious infection"
            ],
            
            # Gastrointestinal
            "abdominal_pain": [
                "Risk of appendicitis requiring surgery",
                "Potential for gastrointestinal complications",
                "May indicate serious abdominal condition"
            ],
            "nausea_vomiting": [
                "Risk of dehydration",
                "Potential for electrolyte imbalance",
                "May indicate serious gastrointestinal condition"
            ],
            "diarrhea": [
                "Risk of dehydration",
                "Potential for electrolyte imbalance",
                "May indicate serious infection"
            ],
            "constipation": [
                "Risk of bowel obstruction",
                "Potential for complications",
                "May affect quality of life"
            ],
            "bloating": [
                "Risk of underlying digestive disorder",
                "Potential for discomfort",
                "May affect nutrition"
            ],
            "heartburn": [
                "Risk of esophageal damage",
                "Potential for Barrett's esophagus",
                "May affect quality of life"
            ],
            "blood_stool": [
                "Risk of serious gastrointestinal condition",
                "Potential for colon cancer",
                "May indicate life-threatening condition"
            ],
            
            # Urinary
            "frequent_urination": [
                "Risk of underlying diabetes",
                "Potential for urinary tract complications",
                "May affect quality of life"
            ],
            "painful_urination": [
                "Risk of kidney infection",
                "Potential for sepsis",
                "May indicate serious infection"
            ],
            "blood_urine": [
                "Risk of serious kidney or bladder condition",
                "Potential for cancer",
                "May indicate life-threatening condition"
            ],
            "incontinence": [
                "Risk of skin irritation",
                "Potential for urinary tract infections",
                "May affect quality of life"
            ],
            
            # Skin
            "rash": [
                "Risk of infection if scratched",
                "Potential for allergic reaction progression",
                "May indicate underlying systemic condition"
            ],
            "itching": [
                "Risk of skin damage from scratching",
                "Potential for infection",
                "May indicate underlying condition"
            ],
            "wound": [
                "Risk of infection",
                "Potential for delayed healing",
                "May require medical attention"
            ],
            "bruising": [
                "Risk of underlying blood disorder",
                "Potential for serious bleeding",
                "May indicate serious condition"
            ],
            "swelling": [
                "Risk of infection",
                "Potential for underlying serious condition",
                "May affect function"
            ],
            "discoloration": [
                "Risk of underlying serious condition",
                "Potential for liver or kidney disease",
                "May indicate life-threatening condition"
            ],
            
            # Musculoskeletal
            "joint_pain": [
                "Risk of decreased mobility",
                "Potential for chronic arthritis",
                "May indicate autoimmune condition"
            ],
            "back_pain": [
                "Risk of chronic pain",
                "Potential for nerve damage",
                "May indicate serious spinal condition"
            ],
            "neck_pain": [
                "Risk of chronic pain",
                "Potential for nerve compression",
                "May affect daily activities"
            ],
            "muscle_pain": [
                "Risk of decreased mobility",
                "Potential for chronic condition",
                "May affect quality of life"
            ],
            "limb_pain": [
                "Risk of decreased function",
                "Potential for chronic pain",
                "May indicate serious condition"
            ],
            
            # General
            "fever": [
                "Risk of complications from infection",
                "Potential for sepsis in severe cases",
                "May indicate serious underlying condition"
            ],
            "fatigue": [
                "Risk of decreased quality of life",
                "Potential for underlying chronic condition",
                "May indicate serious systemic disease"
            ],
            "weight_loss": [
                "Risk of malnutrition",
                "Potential for underlying serious condition",
                "May indicate cancer or chronic disease"
            ],
            "weight_gain": [
                "Risk of obesity-related conditions",
                "Potential for diabetes and heart disease",
                "May affect overall health"
            ],
            "appetite_loss": [
                "Risk of malnutrition",
                "Potential for weight loss",
                "May indicate serious underlying condition"
            ],
            "excessive_thirst": [
                "Risk of diabetes",
                "Potential for dehydration if not addressed",
                "May indicate serious metabolic condition"
            ],
            "night_sweats": [
                "Risk of underlying infection or condition",
                "Potential for sleep disruption",
                "May indicate serious underlying disease"
            ],
            
            # Mental health
            "anxiety": [
                "Risk of panic attacks",
                "Potential for decreased quality of life",
                "May affect daily functioning"
            ],
            "depression": [
                "Risk of suicidal thoughts",
                "Potential for severe depression",
                "May require professional treatment"
            ],
            "insomnia": [
                "Risk of fatigue and decreased function",
                "Potential for mental health issues",
                "May affect overall health"
            ],
            "mood_changes": [
                "Risk of mental health conditions",
                "Potential for relationship problems",
                "May require professional evaluation"
            ],
            
            # Reproductive
            "pelvic_pain": [
                "Risk of underlying serious condition",
                "Potential for fertility issues",
                "May require medical evaluation"
            ],
            "menstrual_irregularities": [
                "Risk of underlying hormonal condition",
                "Potential for fertility issues",
                "May indicate serious condition"
            ],
            "erectile_dysfunction": [
                "Risk of underlying cardiovascular disease",
                "Potential for relationship problems",
                "May indicate serious health condition"
            ],
            
            # Other
            "bleeding": [
                "Risk of excessive blood loss",
                "Potential for life-threatening hemorrhage",
                "May require immediate medical attention"
            ],
            "fainting": [
                "Risk of injury from fall",
                "Potential for underlying serious condition",
                "May indicate life-threatening condition"
            ],
            "tremor": [
                "Risk of decreased function",
                "Potential for underlying neurological condition",
                "May affect daily activities"
            ],
            "balance_problems": [
                "Risk of falls and injury",
                "Potential for serious underlying condition",
                "May indicate neurological condition"
            ],
        }
    
    def _normalize_symptoms(self, symptoms_text: str) -> str:
        """Normalize symptoms text for analysis"""
        return symptoms_text.lower().strip()
    
    def _match_symptoms(self, symptoms_text: str) -> List[str]:
        """Match symptoms text against known patterns"""
        normalized = self._normalize_symptoms(symptoms_text)
        matched_symptoms = []
        
        for symptom_key, patterns in self.symptom_patterns.items():
            for pattern in patterns:
                if pattern in normalized:
                    matched_symptoms.append(symptom_key)
                    break
        
        return matched_symptoms
    
    def _determine_urgency(self, matched_symptoms: List[str]) -> UrgencyLevel:
        """Determine overall urgency based on matched symptoms"""
        if not matched_symptoms:
            return UrgencyLevel.ROUTINE
        
        # Get urgency levels for all matched symptoms
        urgency_levels = []
        for symptom in matched_symptoms:
            if symptom in self.urgency_rules:
                urgency_levels.append(self.urgency_rules[symptom])
        
        if not urgency_levels:
            return UrgencyLevel.ROUTINE
        
        # Return the highest urgency level
        urgency_priority = {
            UrgencyLevel.CRITICAL: 4,
            UrgencyLevel.HIGH: 3,
            UrgencyLevel.MODERATE: 2,
            UrgencyLevel.LOW: 1,
            UrgencyLevel.ROUTINE: 0
        }
        
        return max(urgency_levels, key=lambda x: urgency_priority[x])
    
    def _get_causes(self, matched_symptoms: List[str]) -> List[str]:
        """Get possible causes for matched symptoms"""
        all_causes = []
        
        for symptom in matched_symptoms:
            if symptom in self.cause_database:
                all_causes.extend(self.cause_database[symptom])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_causes = []
        for cause in all_causes:
            if cause not in seen:
                seen.add(cause)
                unique_causes.append(cause)
        
        return unique_causes if unique_causes else ["Unable to determine specific causes"]
    
    def _get_risks(self, matched_symptoms: List[str]) -> List[str]:
        """Get possible risks for matched symptoms"""
        all_risks = []
        
        for symptom in matched_symptoms:
            if symptom in self.risk_factors:
                all_risks.extend(self.risk_factors[symptom])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_risks = []
        for risk in all_risks:
            if risk not in seen:
                seen.add(risk)
                unique_risks.append(risk)
        
        return unique_risks if unique_risks else ["General health monitoring recommended"]
    
    def _initialize_explanations_database(self) -> Dict[str, Dict[str, str]]:
        """Initialize database of simple explanations for medical terms"""
        return {
            "symptom_explanations": {
                # Cardiovascular
                "chest_pain": "Chest pain is any discomfort or pain you feel in your chest area. It can feel like pressure, tightness, or a squeezing sensation. While it can be caused by simple things like muscle strain, it can also be a sign of serious heart or lung problems.",
                "shortness_breath": "Shortness of breath means you're having trouble breathing or feel like you can't get enough air. You might feel like you need to breathe faster or deeper. This can happen during activity or even when resting, and it's important to pay attention to when it occurs.",
                "palpitations": "Palpitations are when you feel your heart beating unusually - it might feel like it's racing, pounding, or skipping beats. While often harmless, they can sometimes indicate a heart problem.",
                "swelling_legs": "Swelling in your legs means fluid is building up, making your legs look puffy or feel heavy. This can be from simple causes like standing too long, or more serious conditions like heart or kidney problems.",
                "high_blood_pressure": "High blood pressure means the force of blood against your artery walls is too high. It often has no symptoms but can lead to serious problems like heart attack or stroke if untreated.",
                "low_blood_pressure": "Low blood pressure means your blood pressure is lower than normal, which can cause dizziness, especially when standing up. It can be harmless or indicate an underlying problem.",
                
                # Neurological
                "severe_headache": "A severe headache is an intense pain in your head that's worse than a normal headache. If it comes on suddenly and is the worst headache you've ever had, it needs immediate attention as it could indicate a serious problem.",
                "dizziness": "Dizziness is when you feel lightheaded, unsteady, or like the room is spinning. It can make you feel like you might faint. This can happen for many reasons, from simple dehydration to more serious conditions.",
                "numbness": "Numbness is when you lose feeling or have reduced sensation in part of your body. You might also feel tingling or 'pins and needles.' This can be from pressure on a nerve or more serious conditions.",
                "seizure": "A seizure is a sudden, uncontrolled electrical disturbance in the brain that can cause changes in behavior, movements, or consciousness. It's a medical emergency that needs immediate attention.",
                "confusion": "Confusion means you're having trouble thinking clearly, remembering things, or understanding what's happening around you. This can be sudden or gradual and may indicate a serious problem.",
                "vision_problems": "Vision problems include blurred vision, double vision, seeing spots, or loss of vision. These can be from eye problems, migraines, or more serious conditions like stroke.",
                "hearing_loss": "Hearing loss means you can't hear as well as usual. You might also hear ringing in your ears (tinnitus). This can be gradual or sudden and may indicate various conditions.",
                "speech_problems": "Speech problems include slurred speech, difficulty finding words, or trouble speaking clearly. This can be a sign of stroke or other serious neurological conditions.",
                "weakness": "Weakness means you have less strength than normal, making it hard to move parts of your body. This can be from muscle problems, nerve damage, or more serious conditions.",
                
                # Respiratory
                "cough": "A cough is your body's way of clearing your throat and airways. While most coughs are from colds or allergies, a persistent or severe cough can indicate more serious problems.",
                "wheezing": "Wheezing is a whistling or squeaky sound when you breathe. It usually means your airways are narrowed, often from asthma, allergies, or other lung conditions.",
                "congestion": "Congestion means your nose is stuffy or blocked, making it hard to breathe through your nose. This is common with colds, allergies, or sinus problems.",
                "sore_throat": "A sore throat is pain, scratchiness, or irritation in your throat that often worsens when you swallow. It's usually from infections but can have other causes.",
                
                # Gastrointestinal
                "abdominal_pain": "Abdominal pain is any discomfort in your belly or stomach area. It can range from a mild ache to severe pain. The location and type of pain can help doctors figure out what's causing it.",
                "nausea_vomiting": "Nausea is feeling like you need to throw up, and vomiting is actually throwing up. This is your body's way of getting rid of something that's bothering it, but it can also lead to dehydration if it continues.",
                "diarrhea": "Diarrhea means you're having loose, watery stools more frequently than normal. It can lead to dehydration and is often from infections, food, or digestive problems.",
                "constipation": "Constipation means you're having difficulty passing stools or going less often than usual. Your stools may be hard and difficult to pass.",
                "bloating": "Bloating is when your belly feels full, tight, or swollen, often with gas. It can be uncomfortable and is usually from digestive issues.",
                "heartburn": "Heartburn is a burning sensation in your chest, often after eating. It's caused by stomach acid flowing back up into your esophagus.",
                "blood_stool": "Blood in your stool means there's bleeding somewhere in your digestive tract. It can appear as red blood or make stools look black and tarry. This always needs medical attention.",
                
                # Urinary
                "frequent_urination": "Frequent urination means you need to urinate more often than usual. This can be from drinking more fluids, infections, diabetes, or other conditions.",
                "painful_urination": "Painful urination means it hurts or burns when you urinate. This is often a sign of a urinary tract infection or other urinary problem.",
                "blood_urine": "Blood in your urine means there's bleeding in your urinary tract. Your urine might look pink, red, or brown. This always needs medical attention.",
                "incontinence": "Incontinence means you can't control when you urinate, leading to accidental leakage. This can be from various causes and affects quality of life.",
                
                # Skin
                "rash": "A rash is any change in your skin's appearance - it might be red, bumpy, itchy, or scaly. Rashes can be caused by allergies, infections, or other skin conditions.",
                "itching": "Itching is an uncomfortable sensation that makes you want to scratch. It can be from dry skin, allergies, infections, or underlying medical conditions.",
                "wound": "A wound is any break in your skin, from a small cut to a larger injury. Proper care is important to prevent infection and promote healing.",
                "bruising": "Bruising is when blood collects under your skin after an injury, causing discoloration. Easy or excessive bruising may indicate a problem.",
                "swelling": "Swelling is when part of your body becomes larger due to fluid buildup or inflammation. It can be from injury, infection, or underlying conditions.",
                "discoloration": "Skin discoloration means your skin color has changed - it might be yellow (jaundice), pale, or have other color changes. This can indicate various conditions.",
                
                # Musculoskeletal
                "joint_pain": "Joint pain is discomfort in any of your body's joints (where bones meet, like knees, elbows, wrists). It can be caused by injury, overuse, or conditions like arthritis.",
                "back_pain": "Back pain is discomfort in your back, often in the lower back area. It's very common and can be caused by muscle strain, poor posture, or more serious spinal conditions.",
                "neck_pain": "Neck pain is discomfort in your neck area, often from muscle strain, poor posture, or more serious conditions affecting the spine.",
                "muscle_pain": "Muscle pain is discomfort in your muscles, often from overuse, injury, or conditions like fibromyalgia.",
                "limb_pain": "Limb pain is discomfort in your arms or legs, which can be from injury, nerve problems, circulation issues, or other conditions.",
                
                # General
                "fever": "A fever is when your body temperature is higher than normal (usually above 100.4°F or 38°C). It's your body's way of fighting off infections. A mild fever is usually okay, but a high fever or one that lasts several days needs attention.",
                "fatigue": "Fatigue is extreme tiredness that doesn't go away with rest. It's more than just being sleepy - it's a persistent lack of energy that can affect your daily activities.",
                "weight_loss": "Weight loss means you're losing weight without trying. While sometimes intentional, unexplained weight loss can indicate a serious condition.",
                "weight_gain": "Weight gain means you're putting on weight, which can be from diet, lifestyle, or underlying medical conditions.",
                "appetite_loss": "Loss of appetite means you don't feel hungry or don't want to eat. This can be from various causes, including illness, stress, or medical conditions.",
                "excessive_thirst": "Excessive thirst means you're unusually thirsty and drinking much more than normal. This can be a sign of diabetes or other conditions.",
                "night_sweats": "Night sweats are episodes of excessive sweating during sleep that soak your clothes and bedding. They can be from various causes, including infections or hormonal changes.",
                
                # Mental health
                "anxiety": "Anxiety is feeling worried, nervous, or uneasy, often about something with an uncertain outcome. While normal in some situations, excessive anxiety can interfere with daily life.",
                "depression": "Depression is more than just feeling sad - it's a persistent feeling of sadness, hopelessness, or loss of interest that affects daily life and requires treatment.",
                "insomnia": "Insomnia means you have trouble falling asleep, staying asleep, or getting restful sleep. This can affect your energy, mood, and overall health.",
                "mood_changes": "Mood changes are shifts in how you feel emotionally - you might feel happy one moment and sad the next, or be more irritable than usual.",
                
                # Reproductive
                "pelvic_pain": "Pelvic pain is discomfort in the lower abdomen or pelvic area. It can be from various causes including menstrual issues, infections, or other conditions.",
                "menstrual_irregularities": "Menstrual irregularities include changes in your period - it might be too frequent, too infrequent, too heavy, or missed entirely. This can indicate various conditions.",
                "erectile_dysfunction": "Erectile dysfunction is the inability to get or maintain an erection sufficient for sexual activity. It can be from physical or psychological causes.",
                
                # Other
                "bleeding": "Bleeding is when blood escapes from your blood vessels. While normal from injuries, uncontrolled or excessive bleeding needs immediate medical attention.",
                "fainting": "Fainting (syncope) is a temporary loss of consciousness, usually from a drop in blood pressure or lack of oxygen to the brain. It can indicate various conditions.",
                "tremor": "A tremor is an involuntary shaking or trembling movement, often in your hands. It can be from various causes including neurological conditions.",
                "balance_problems": "Balance problems mean you feel unsteady or have trouble maintaining your balance when standing or walking. This increases your risk of falling.",
            },
            "cause_explanations": {
                # Cardiovascular causes
                "Heart attack or angina": "This happens when blood flow to your heart is blocked. It's a medical emergency. Angina is chest pain from reduced blood flow, while a heart attack means the heart muscle is being damaged.",
                "Pulmonary embolism": "This is when a blood clot blocks an artery in your lungs. It's serious and can be life-threatening, causing sudden shortness of breath and chest pain.",
                "Costochondritis": "This is inflammation of the cartilage that connects your ribs to your breastbone. It causes chest pain that can feel like a heart problem but is usually harmless.",
                "Pericarditis": "This is inflammation of the sac around your heart. It causes chest pain and can be from infections or other causes.",
                "Heart failure": "This means your heart isn't pumping blood as well as it should. It can cause shortness of breath, fatigue, and swelling.",
                "Pulmonary hypertension": "This is high blood pressure in the arteries of your lungs. It makes your heart work harder and can cause shortness of breath.",
                "Arrhythmia": "This is an irregular heartbeat - your heart might beat too fast, too slow, or irregularly. Some are harmless, others need treatment.",
                "Hyperthyroidism": "This is when your thyroid gland produces too much hormone, speeding up your metabolism. It can cause palpitations, weight loss, and anxiety.",
                "Venous insufficiency": "This means the veins in your legs aren't working properly to return blood to your heart. It can cause leg swelling.",
                "Deep vein thrombosis (DVT)": "This is a blood clot in a deep vein, usually in your leg. It's serious because the clot can travel to your lungs.",
                "Lymphedema": "This is swelling caused by a problem with your lymphatic system, which helps drain fluid from tissues.",
                "Essential hypertension": "This is high blood pressure with no known cause. It's the most common type and usually develops over time.",
                "Postural hypotension": "This is low blood pressure that happens when you stand up quickly, causing dizziness.",
                
                # Neurological causes
                "Migraine": "This is a severe type of headache that can cause intense pain, nausea, and sensitivity to light. It can last for hours or even days.",
                "Tension headache": "This is the most common type of headache, usually from stress or muscle tension. It feels like a tight band around your head.",
                "Cluster headache": "This is a very painful type of headache that occurs in clusters or groups. The pain is usually around one eye.",
                "Meningitis": "This is inflammation of the membranes around your brain and spinal cord. It's serious and can be life-threatening.",
                "Brain tumor (rare)": "This is an abnormal growth in your brain. While rare, it can cause headaches and other neurological symptoms.",
                "Stroke (rare)": "This happens when blood flow to part of your brain is cut off. It's a medical emergency that can cause permanent damage.",
                "Benign paroxysmal positional vertigo (BPPV)": "This is a common cause of dizziness. It happens when tiny crystals in your inner ear get dislodged.",
                "Meniere's disease": "This is an inner ear disorder that causes episodes of vertigo, hearing loss, and ringing in the ears.",
                "Pinched nerve": "This happens when surrounding tissues press on a nerve, causing pain, numbness, or tingling.",
                "Multiple sclerosis": "This is a disease where your immune system attacks the protective covering of your nerves, causing various neurological symptoms.",
                "Stroke": "This happens when blood flow to part of your brain is cut off, causing brain damage. It's a medical emergency.",
                "Transient ischemic attack (TIA)": "This is a 'mini-stroke' - a temporary blockage of blood flow to the brain. It's a warning sign of a possible future stroke.",
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
                "Age-related hearing loss": "This is gradual hearing loss that happens as you get older. It's common and can be helped with hearing aids.",
                "Acoustic neuroma (rare)": "This is a noncancerous tumor on the nerve that connects your ear to your brain. It's rare but can cause hearing loss.",
                "Guillain-Barré syndrome": "This is a rare disorder where your immune system attacks your nerves, causing weakness and sometimes paralysis.",
                "Parkinson's disease": "This is a progressive nervous system disorder that affects movement, causing tremors, stiffness, and balance problems.",
                "Essential tremor": "This is a neurological disorder that causes involuntary shaking, usually in your hands. It's different from Parkinson's.",
                "Alcohol withdrawal": "This happens when someone who drinks heavily suddenly stops, causing tremors, anxiety, and other symptoms.",
                
                # Respiratory causes
                "Pneumonia": "This is an infection in your lungs that makes it hard to breathe. It can be caused by bacteria or viruses and often comes with fever and cough.",
                "Asthma": "This is a condition where your airways become narrow and swollen, making it hard to breathe. It can be triggered by allergies, exercise, or other factors.",
                "Chronic obstructive pulmonary disease (COPD)": "This is a long-term lung disease that makes it hard to breathe. It's usually caused by smoking and gets worse over time.",
                "Bronchitis": "This is inflammation of the tubes that carry air to your lungs. It causes a persistent cough, often with mucus. It can be acute (short-term) or chronic (long-term).",
                "Sinusitis": "This is inflammation of your sinuses, the air-filled spaces in your face. It causes congestion, facial pain, and sometimes headaches.",
                "Deviated septum": "This means the wall between your nostrils is crooked, which can cause congestion and breathing problems.",
                "Nasal polyps": "These are soft, noncancerous growths in your nose that can cause congestion and breathing problems.",
                "Post-nasal drip": "This is when mucus from your nose drips down the back of your throat, causing cough and throat irritation.",
                
                # Gastrointestinal causes
                "Gastroenteritis": "This is inflammation of your stomach and intestines, often called 'stomach flu.' It causes nausea, vomiting, diarrhea, and stomach pain.",
                "Appendicitis": "This is when your appendix (a small organ in your lower right abdomen) becomes inflamed. It causes severe abdominal pain and usually needs surgery.",
                "Gallstones": "These are hard deposits that form in your gallbladder. They can cause abdominal pain, especially after eating fatty foods.",
                "Irritable bowel syndrome (IBS)": "This is a common disorder affecting the large intestine, causing cramping, abdominal pain, bloating, gas, diarrhea, and constipation.",
                "Food poisoning": "This happens when you eat food contaminated with bacteria or viruses. It causes nausea, vomiting, diarrhea, and stomach pain, usually within hours of eating.",
                "Constipation": "This means you're having difficulty passing stools or going less often than usual. Your stools may be hard and difficult to pass.",
                "Ulcer": "This is a sore in the lining of your stomach or small intestine. It can cause abdominal pain, especially when your stomach is empty.",
                "Diverticulitis": "This is inflammation of small pouches in your colon. It causes abdominal pain, usually on the left side, and can be serious.",
                "Inflammatory bowel disease (IBD)": "This includes Crohn's disease and ulcerative colitis - chronic conditions that cause inflammation in your digestive tract.",
                "Lactose intolerance": "This means your body can't digest lactose, a sugar in milk. It causes bloating, gas, and diarrhea after eating dairy.",
                "Celiac disease": "This is an immune reaction to eating gluten, a protein in wheat. It damages your small intestine and causes digestive symptoms.",
                "Hiatal hernia": "This is when part of your stomach pushes up through your diaphragm, which can cause heartburn and GERD.",
                
                # Urinary causes
                "Urinary tract infection (UTI)": "This is an infection in any part of your urinary system - kidneys, bladder, or urethra. It causes painful urination and frequent urination.",
                "Bladder infection": "This is a type of UTI specifically in your bladder. It causes painful and frequent urination.",
                "Kidney infection": "This is a serious UTI that has spread to your kidneys. It causes back pain, fever, and can be serious if untreated.",
                "Kidney stones": "These are hard deposits that form in your kidneys. They can cause severe pain when they pass through your urinary tract.",
                "Overactive bladder": "This means your bladder contracts too often, causing frequent and urgent urination.",
                "Prostate problems": "The prostate is a gland in men. When it's enlarged or infected, it can cause urinary problems.",
                "Interstitial cystitis": "This is a chronic condition causing bladder pressure and pain, along with frequent, painful urination.",
                "Sexually transmitted infection (STI)": "These are infections passed through sexual contact. Some can cause painful urination.",
                
                # Skin causes
                "Allergic reaction": "This happens when your immune system overreacts to something harmless (like pollen, food, or medication). It can cause rashes, swelling, or more serious symptoms.",
                "Eczema": "This is a skin condition that causes dry, itchy, and inflamed skin. It's common and can be managed with proper skin care and medication.",
                "Psoriasis": "This is a skin condition that causes red, scaly patches. It's an autoimmune condition that can be managed with treatment.",
                "Contact dermatitis": "This is a rash caused by touching something that irritates your skin or causes an allergic reaction.",
                "Fungal infection": "This is an infection caused by fungi, like athlete's foot or ringworm. It can cause rashes and itching.",
                "Parasitic infection": "This is when parasites (like scabies or lice) infest your skin, causing itching and rashes.",
                
                # Musculoskeletal causes
                "Arthritis": "This is inflammation of your joints, causing pain, stiffness, and swelling. There are many types, with osteoarthritis and rheumatoid arthritis being most common.",
                "Osteoarthritis": "This is the most common type of arthritis, caused by wear and tear on your joints over time.",
                "Rheumatoid arthritis": "This is an autoimmune disease that causes inflammation in your joints, leading to pain and deformity.",
                "Gout": "This is a type of arthritis caused by uric acid crystals in your joints, causing sudden, severe pain, often in your big toe.",
                "Injury": "This is damage to your body from an accident or trauma. It can cause pain, swelling, and limited movement.",
                "Overuse": "This is pain from repetitive movements or overworking a muscle or joint. Rest usually helps.",
                "Lupus": "This is an autoimmune disease that can affect many parts of your body, including joints, skin, and organs.",
                "Fibromyalgia": "This is a condition causing widespread muscle pain, fatigue, and tender points throughout your body.",
                "Sciatica": "This is pain along the sciatic nerve, which runs from your lower back down your leg. It's often from a herniated disc.",
                "Spinal stenosis": "This is narrowing of the spaces in your spine, which can press on nerves and cause pain.",
                "Osteoporosis": "This is a condition where your bones become weak and brittle, increasing fracture risk.",
                "Whiplash": "This is a neck injury from a sudden back-and-forth movement, often from car accidents.",
                
                # General causes
                "Viral infection (flu, cold)": "These are common illnesses caused by viruses. They usually cause fever, body aches, and fatigue, and most people recover with rest and fluids.",
                "Bacterial infection": "This is when harmful bacteria get into your body and cause illness. These often need antibiotics to treat and can cause fever and other symptoms.",
                "Urinary tract infection": "This is an infection in any part of your urinary system. It can cause fever, pain, and frequent urination.",
                "Respiratory infection": "This is an infection in your respiratory system - your nose, throat, or lungs. It can cause cough, congestion, and fever.",
                "Autoimmune condition": "This is when your immune system mistakenly attacks your own body. There are many types, like rheumatoid arthritis and lupus.",
                "Medication reaction": "This is when a medication causes an unwanted side effect or allergic reaction.",
                "Cancer (rare)": "This is when cells in your body grow uncontrollably. While it can cause various symptoms, most symptoms are from other causes.",
                "Sleep apnea": "This is when you stop breathing repeatedly during sleep. It causes fatigue and can lead to serious health problems.",
                "Chronic fatigue syndrome": "This is a complex disorder causing extreme fatigue that doesn't improve with rest and can't be explained by an underlying medical condition.",
                "Diabetes": "This is when your body can't properly use or produce insulin, causing high blood sugar. It can cause many symptoms including excessive thirst, frequent urination, and fatigue.",
                "Diabetes insipidus": "This is a rare condition causing excessive thirst and urination, but it's different from diabetes mellitus.",
                "Cushing's syndrome": "This is when your body has too much cortisol, a stress hormone. It can cause weight gain and other symptoms.",
                "Hyperthyroidism": "This is when your thyroid produces too much hormone, speeding up your metabolism.",
                "Hypothyroidism": "This is when your thyroid doesn't produce enough hormone, slowing down your metabolism.",
                "Menopause": "This is when a woman's periods stop permanently, usually around age 50. It can cause night sweats, mood changes, and other symptoms.",
                "Perimenopause": "This is the time before menopause when hormone levels start changing. It can cause irregular periods and other symptoms.",
                "Polycystic ovary syndrome (PCOS)": "This is a hormonal disorder in women that can cause irregular periods, weight gain, and other symptoms.",
                "Endometriosis": "This is when tissue similar to the lining of your uterus grows outside it, causing pelvic pain and other symptoms.",
                "Ovarian cysts": "These are fluid-filled sacs in or on your ovaries. Most are harmless, but some can cause pain.",
                "Pelvic inflammatory disease (PID)": "This is an infection of the female reproductive organs, often from sexually transmitted bacteria.",
                "Vasovagal syncope": "This is fainting caused by a sudden drop in heart rate and blood pressure, often from stress or pain.",
                "Restless leg syndrome": "This is a condition causing an uncontrollable urge to move your legs, often at night, disrupting sleep.",
                
                # Mental health causes
                "Generalized anxiety disorder": "This is excessive worry about everyday things that's hard to control and interferes with daily life.",
                "Panic disorder": "This causes sudden, intense episodes of fear (panic attacks) with physical symptoms like rapid heartbeat and shortness of breath.",
                "Major depressive disorder": "This is persistent sadness and loss of interest that affects how you feel, think, and handle daily activities.",
                "Bipolar disorder": "This causes extreme mood swings from emotional highs (mania) to lows (depression).",
                "Stress": "This is your body's response to pressure or demands. While normal, excessive stress can affect your health.",
                "Substance use": "Using drugs or alcohol can cause various physical and mental health symptoms.",
                "Poor sleep habits": "This includes irregular sleep schedules, using screens before bed, or other habits that interfere with good sleep.",
            },
            "risk_explanations": {
                "Risk of heart attack or cardiac event": "This means there's a chance of a serious heart problem that could be life-threatening. Immediate medical attention is crucial.",
                "Risk of respiratory failure": "This means your breathing could become so difficult that you need help breathing. This is serious and needs immediate care.",
                "Risk of stroke or brain hemorrhage": "This means there's a chance of a serious brain problem. A stroke happens when blood flow to the brain is cut off, and a brain hemorrhage is bleeding in the brain.",
                "Risk of complications from infection": "This means an infection could get worse or spread, potentially causing more serious health problems. Proper treatment is important.",
                "Risk of appendicitis requiring surgery": "This means your appendix might be infected and need to be removed. This is a common surgery and usually goes well.",
                "Risk of falls and injury": "This means you're more likely to fall and hurt yourself, especially if you feel dizzy or unsteady. Be careful when moving around.",
                "Risk of dehydration": "This means your body could lose too much water, especially if you're vomiting or have diarrhea. Make sure to drink fluids.",
                "Risk of infection if scratched": "If you scratch a rash, you could break the skin and let bacteria in, causing an infection. Try not to scratch.",
                "Risk of decreased quality of life": "This means your symptoms could make daily activities harder and affect how you feel overall. Getting treatment can help.",
                "Risk of chronic pain": "This means the pain could become long-lasting and affect your daily life. Early treatment can help prevent this.",
                "Risk of decreased mobility": "This means you might have trouble moving around, which can affect your independence and daily activities."
            }
        }
    
    def _explain_symptom(self, symptom_key: str) -> str:
        """Get simple explanation for a symptom"""
        explanations = self.explanations_database.get("symptom_explanations", {})
        return explanations.get(symptom_key, f"{symptom_key.replace('_', ' ').title()} is a symptom that should be evaluated by a healthcare provider.")
    
    def _explain_cause(self, cause: str) -> str:
        """Get simple explanation for a cause"""
        explanations = self.explanations_database.get("cause_explanations", {})
        return explanations.get(cause, f"{cause} is a possible cause that a doctor can help diagnose and treat.")
    
    def _explain_risk(self, risk: str) -> str:
        """Get simple explanation for a risk"""
        explanations = self.explanations_database.get("risk_explanations", {})
        return explanations.get(risk, f"{risk} is something to be aware of and discuss with your healthcare provider.")
    
    def _generate_simplified_explanations(self, matched_symptoms: List[str], 
                                         causes: List[str], 
                                         risks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Generate simplified explanations for symptoms, causes, and risks"""
        symptom_explanations = []
        for symptom in matched_symptoms:
            symptom_explanations.append({
                "term": symptom.replace('_', ' ').title(),
                "explanation": self._explain_symptom(symptom)
            })
        
        cause_explanations = []
        for cause in causes:
            cause_explanations.append({
                "term": cause,
                "explanation": self._explain_cause(cause)
            })
        
        risk_explanations = []
        for risk in risks:
            risk_explanations.append({
                "term": risk,
                "explanation": self._explain_risk(risk)
            })
        
        return {
            "symptom_explanations": symptom_explanations,
            "cause_explanations": cause_explanations,
            "risk_explanations": risk_explanations
        }
    
    def diagnose(self, symptoms_text: str, patient_age: int = None, 
                 additional_info: str = None) -> Dict:
        """
        Main diagnosis function
        
        Args:
            symptoms_text: Description of symptoms
            patient_age: Age of patient (optional)
            additional_info: Additional medical information (optional)
        
        Returns:
            Dictionary containing urgency, causes, risks, and recommendations
        """
        try:
            # Match symptoms
            matched_symptoms = self._match_symptoms(symptoms_text)
            
            # Determine urgency
            urgency = self._determine_urgency(matched_symptoms)
            
            # Get causes
            causes = self._get_causes(matched_symptoms)
            
            # Get risks
            risks = self._get_risks(matched_symptoms)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                urgency, matched_symptoms, patient_age, additional_info
            )
            
            # Generate simplified explanations
            explanations = self._generate_simplified_explanations(matched_symptoms, causes, risks)
            
            return {
                "urgency_level": urgency.value,
                "urgency_priority": urgency.name,
                "matched_symptoms": matched_symptoms,
                "possible_causes": causes,
                "possible_risks": risks,
                "recommendations": recommendations,
                "matched_symptom_count": len(matched_symptoms),
                "explanations": explanations
            }
            
        except Exception as e:
            return {
                "error": f"Error in diagnosis: {str(e)}",
                "urgency_level": "Unable to determine",
                "possible_causes": [],
                "possible_risks": []
            }
    
    def _generate_recommendations(self, urgency: UrgencyLevel, 
                                  matched_symptoms: List[str],
                                  patient_age: int = None,
                                  additional_info: str = None) -> List[str]:
        """Generate personalized recommendations based on diagnosis"""
        recommendations = []
        
        # Urgency-based recommendations
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
        
        # Age-based recommendations
        if patient_age:
            if patient_age < 18:
                recommendations.append("Note: Pediatric symptoms may require specialized care")
            elif patient_age > 65:
                recommendations.append("Note: Elderly patients may need closer monitoring")
        
        # General recommendations
        recommendations.append("Keep a symptom diary to track changes")
        recommendations.append("Stay hydrated and get adequate rest")
        recommendations.append("⚠️ This is not a substitute for professional medical advice")
        
        return recommendations


def diagnose_symptoms(symptoms_text: str, patient_age: int = None, 
                     additional_info: str = None) -> Dict:
    """
    Convenience function for symptoms diagnosis
    Now uses ML-based diagnosis instead of rule-based
    
    Args:
        symptoms_text: Description of symptoms
        patient_age: Age of patient (optional)
        additional_info: Additional medical information (optional)
    
    Returns:
        Dictionary containing diagnosis results
    """
    # Use enhanced ML-based diagnosis (preferred) or fallback to standard ML
    try:
        from .ml_diagnosis_enhanced import get_enhanced_ml_model
        model = get_enhanced_ml_model()
        full_text = symptoms_text
        if additional_info:
            full_text += " " + additional_info
        return model.predict(full_text, patient_age)
    except (ImportError, ValueError) as e:
        # Fallback to standard ML diagnosis
        try:
            from .ml_diagnosis import diagnose_symptoms_ml
            return diagnose_symptoms_ml(symptoms_text, patient_age, additional_info)
        except (ImportError, ValueError) as e2:
            # Fallback to rule-based if ML fails or models not trained
            try:
                agent = SymptomsDiagnosisAgent()
                return agent.diagnose(symptoms_text, patient_age, additional_info)
            except Exception as fallback_error:
                return {
                    "error": f"ML diagnosis failed: {e2}. Rule-based fallback also failed: {fallback_error}",
                    "urgency_level": "Unable to determine",
                    "possible_causes": [],
                    "possible_risks": [],
                    "recommendations": ["Please train the ML models first using: python train_models.py"]
                }

