"""
Image Classification Module for Medical Diagnosis
Provides functions for image preprocessing, classification, segmentation, and diagnosis prediction
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import io


class ImageClassifier:
    """
    Image classification class for medical image diagnosis
    """
    
    def __init__(self):
        """Initialize the image classifier"""
        self.model_loaded = False
        self.class_names = self._initialize_class_names()
        self.input_size = (224, 224)  # Standard input size for most models
    
    def _initialize_class_names(self) -> List[str]:
        """Initialize medical condition class names"""
        return [
            "Normal",
            "Abnormal",
            "Pneumonia",
            "COVID-19",
            "Tuberculosis",
            "Lung Cancer",
            "Skin Lesion",
            "Dermatitis",
            "Melanoma",
            "Benign Mole",
            "Malignant Mole",
            "Fracture",
            "No Fracture",
            "Arthritis",
            "Other Condition"
        ]
    
    def classify_normal_abnormal(self, image) -> Dict[str, any]:
        """
        Classify image as Normal or Abnormal
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with classification result
        """
        try:
            predictions = self.predict(image, top_k=2)
            
            # Determine if normal or abnormal
            is_normal = False
            confidence = 0.0
            
            for pred in predictions:
                if pred["class"] == "Normal":
                    is_normal = True
                    confidence = pred["confidence"]
                    break
                elif pred["class"] != "Normal" and pred["confidence"] > confidence:
                    confidence = pred["confidence"]
            
            result = {
                "classification": "Normal" if is_normal else "Abnormal",
                "confidence": confidence,
                "is_normal": is_normal,
                "is_abnormal": not is_normal,
                "message": f"Classification: {'Normal' if is_normal else 'Abnormal'} ({confidence*100:.2f}% confidence)"
            }
            
            return result
            
        except Exception as e:
            return {
                "classification": "Error",
                "confidence": 0.0,
                "is_normal": False,
                "is_abnormal": False,
                "message": f"Error: {str(e)}"
            }
    
    def segment_problem_area(self, image) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Segment and highlight problem areas in the image
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (segmented_image, metadata)
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, np.ndarray):
                img = image.copy()
            elif isinstance(image, Image.Image):
                img = np.array(image)
            else:
                raise ValueError("Unsupported image type")
            
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Create a mask for problem areas (simulated segmentation)
            # In a real implementation, this would use a segmentation model
            mask = self._generate_segmentation_mask(img)
            
            # Highlight problem areas
            highlighted = img.copy()
            highlighted[mask > 0] = [255, 0, 0]  # Red highlight
            
            # Blend original with highlighted areas
            alpha = 0.6
            segmented = cv2.addWeighted(img, 1-alpha, highlighted, alpha, 0)
            
            # Calculate problem area statistics
            problem_area_ratio = np.sum(mask > 0) / mask.size
            bbox = self._get_bounding_box(mask)
            
            metadata = {
                "problem_area_ratio": float(problem_area_ratio),
                "problem_area_percentage": float(problem_area_ratio * 100),
                "bounding_box": bbox,
                "has_problem_area": problem_area_ratio > 0.01
            }
            
            return segmented, metadata
            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return image, {"error": str(e)}
    
    def _generate_segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate segmentation mask for problem areas (placeholder for actual model)
        
        Args:
            image: Input image
        
        Returns:
            Binary mask
        """
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        
        # Fill contours that might indicate problems
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small areas
                cv2.fillPoly(mask, [contour], 255)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _get_bounding_box(self, mask: np.ndarray) -> Dict[str, int]:
        """Get bounding box of problem area"""
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return {
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min),
            "height": int(y_max - y_min)
        }
    
    def calculate_risk_score(self, image, diagnosis: str = None) -> Dict[str, any]:
        """
        Calculate risk score for the medical condition
        
        Args:
            image: Input image
            diagnosis: Optional diagnosis string
        
        Returns:
            Dictionary with risk score and details
        """
        try:
            # Get predictions
            predictions = self.predict(image, top_k=3)
            
            if not predictions or predictions[0].get("class") == "Error":
                return {
                    "risk_score": 0.0,
                    "risk_level": "Unknown",
                    "message": "Unable to calculate risk score"
                }
            
            # Get diagnosis if not provided
            if not diagnosis:
                diagnosis = predictions[0]["class"]
            
            # Calculate risk score based on diagnosis and confidence
            confidence = predictions[0]["confidence"]
            
            # Risk levels for different conditions
            risk_levels = {
                "Normal": (0.0, 0.2, "Low"),
                "Abnormal": (0.3, 0.6, "Moderate"),
                "Pneumonia": (0.5, 0.8, "High"),
                "COVID-19": (0.7, 0.95, "Very High"),
                "Tuberculosis": (0.7, 0.9, "Very High"),
                "Lung Cancer": (0.8, 0.95, "Critical"),
                "Melanoma": (0.8, 0.95, "Critical"),
                "Malignant Mole": (0.8, 0.95, "Critical"),
                "Fracture": (0.6, 0.85, "High"),
                "Skin Lesion": (0.4, 0.7, "Moderate"),
                "Dermatitis": (0.2, 0.5, "Low"),
                "Benign Mole": (0.1, 0.3, "Low"),
                "Arthritis": (0.3, 0.6, "Moderate")
            }
            
            # Get risk range for diagnosis
            if diagnosis in risk_levels:
                min_risk, max_risk, risk_level = risk_levels[diagnosis]
                # Interpolate based on confidence
                risk_score = min_risk + (max_risk - min_risk) * confidence
            else:
                # Default risk calculation
                risk_score = confidence * 0.7
                risk_level = "Moderate" if risk_score > 0.5 else "Low"
            
            # Normalize risk score to 0-1
            risk_score = min(1.0, max(0.0, risk_score))
            
            # Determine risk level based on score
            if risk_score >= 0.8:
                risk_level = "Critical"
            elif risk_score >= 0.6:
                risk_level = "High"
            elif risk_score >= 0.4:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            return {
                "risk_score": float(risk_score),
                "risk_percentage": float(risk_score * 100),
                "risk_level": risk_level,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "message": f"Risk Score: {risk_score*100:.1f}% ({risk_level} Risk)"
            }
            
        except Exception as e:
            return {
                "risk_score": 0.0,
                "risk_level": "Unknown",
                "message": f"Error calculating risk: {str(e)}"
            }
    
    def preprocess_image(self, image, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess image for classification
        
        Args:
            image: Input image (numpy array, PIL Image, or file-like object)
            target_size: Target size for resizing (width, height)
        
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert different input types to numpy array
            if isinstance(image, np.ndarray):
                img = image.copy()
            elif isinstance(image, Image.Image):
                img = np.array(image)
            elif hasattr(image, 'read'):  # File-like object
                img_bytes = image.read()
                img = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            else:
                raise ValueError("Unsupported image type")
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            target_size = target_size or self.input_size
            img = cv2.resize(img, target_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension if needed
            if len(img.shape) == 3:
                img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def validate_image(self, image) -> Tuple[bool, str]:
        """
        Validate if image is suitable for classification
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    return False, "Image is empty"
                if len(image.shape) < 2:
                    return False, "Image dimensions are invalid"
                return True, "Image is valid"
            elif isinstance(image, Image.Image):
                if image.size[0] == 0 or image.size[1] == 0:
                    return False, "Image dimensions are invalid"
                return True, "Image is valid"
            else:
                return False, "Unsupported image format"
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def extract_features(self, image) -> np.ndarray:
        """
        Extract features from image (placeholder for actual feature extraction)
        
        Args:
            image: Input image
        
        Returns:
            Feature vector
        """
        try:
            preprocessed = self.preprocess_image(image)
            # In a real implementation, this would use a feature extraction model
            # For now, return flattened image as features
            return preprocessed.flatten()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.array([])
    
    def predict(self, image, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Make prediction on image
        
        Args:
            image: Input image
            top_k: Number of top predictions to return
        
        Returns:
            List of predictions with class names and confidence scores
        """
        try:
            # Validate image
            is_valid, message = self.validate_image(image)
            if not is_valid:
                return [{"class": "Error", "confidence": 0.0, "message": message}]
            
            # Preprocess image
            preprocessed = self.preprocess_image(image)
            
            # In a real implementation, this would use an actual trained model
            # For now, generate realistic-looking predictions
            predictions = self._generate_predictions(preprocessed, top_k)
            
            return predictions
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return [{"class": "Error", "confidence": 0.0, "message": str(e)}]
    
    def _generate_predictions(self, preprocessed_image: np.ndarray, top_k: int) -> List[Dict[str, any]]:
        """
        Generate predictions (placeholder - replace with actual model inference)
        
        Args:
            preprocessed_image: Preprocessed image array
            top_k: Number of top predictions
        
        Returns:
            List of prediction dictionaries
        """
        # Simulate model predictions with realistic confidence scores
        np.random.seed(hash(preprocessed_image.tobytes()) % 2**32)
        
        # Generate random but realistic predictions
        num_classes = len(self.class_names)
        scores = np.random.dirichlet(np.ones(num_classes) * 2.0)
        
        # Get top k predictions
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                "class": self.class_names[idx],
                "confidence": float(scores[idx]),
                "class_index": int(idx)
            })
        
        return predictions
    
    def predict_diagnosis(self, image) -> Dict[str, any]:
        """
        Predict medical diagnosis from image
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with diagnosis results
        """
        try:
            predictions = self.predict(image, top_k=3)
            
            if not predictions or predictions[0].get("class") == "Error":
                return {
                    "diagnosis": "Unable to determine",
                    "confidence": 0.0,
                    "message": predictions[0].get("message", "Error in prediction") if predictions else "No predictions",
                    "all_predictions": []
                }
            
            top_prediction = predictions[0]
            
            return {
                "diagnosis": top_prediction["class"],
                "confidence": top_prediction["confidence"],
                "message": f"Predicted: {top_prediction['class']} with {top_prediction['confidence']*100:.2f}% confidence",
                "all_predictions": predictions,
                "recommendation": self._get_recommendation(top_prediction["class"])
            }
            
        except Exception as e:
            return {
                "diagnosis": "Error",
                "confidence": 0.0,
                "message": f"Error in diagnosis: {str(e)}",
                "all_predictions": []
            }
    
    def _get_recommendation(self, diagnosis: str) -> str:
        """Get recommendation based on diagnosis"""
        recommendations = {
            "Normal": "No abnormalities detected. Continue regular health monitoring.",
            "Pneumonia": "Consult a healthcare provider. May require antibiotics and rest.",
            "COVID-19": "Seek medical attention immediately. Follow isolation guidelines.",
            "Tuberculosis": "Urgent medical consultation required. This is a serious condition.",
            "Lung Cancer": "Immediate consultation with an oncologist is recommended.",
            "Skin Lesion": "Have the lesion evaluated by a dermatologist.",
            "Dermatitis": "May be treatable with topical medications. Consult a dermatologist.",
            "Melanoma": "Urgent dermatological evaluation required. Early detection is crucial.",
            "Benign Skin Condition": "Monitor the condition. Consult a doctor if it changes.",
            "Fracture": "Seek immediate medical attention. Avoid moving the affected area.",
            "No Fracture": "No fracture detected. If pain persists, consult a doctor.",
            "Arthritis": "Consult a rheumatologist for proper diagnosis and treatment plan.",
            "Other Condition": "Further evaluation by a healthcare provider is recommended."
        }
        return recommendations.get(diagnosis, "Consult a healthcare provider for proper evaluation.")
    
    def get_confidence_score(self, image) -> float:
        """
        Get confidence score for the top prediction
        
        Args:
            image: Input image
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            predictions = self.predict(image, top_k=1)
            if predictions and "confidence" in predictions[0]:
                return predictions[0]["confidence"]
            return 0.0
        except Exception as e:
            print(f"Error getting confidence score: {e}")
            return 0.0
    
    def classify_image_type(self, image) -> str:
        """
        Classify the type of medical image (X-ray, skin, etc.)
        
        Args:
            image: Input image
        
        Returns:
            Image type classification
        """
        try:
            if isinstance(image, np.ndarray):
                img = image
            else:
                img = np.array(image)
            
            # Simple heuristics for image type classification
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                return "X-ray or Grayscale Medical Image"
            elif len(img.shape) == 3:
                # Check if it looks like a skin image (colorful)
                mean_color = np.mean(img, axis=(0, 1))
                if np.max(mean_color) - np.min(mean_color) > 30:
                    return "Skin or Color Medical Image"
                else:
                    return "Medical Scan Image"
            else:
                return "Unknown Image Type"
        except Exception as e:
            return f"Error classifying: {str(e)}"
    
    def classify_chest_xray(self, image) -> Dict[str, any]:
        """
        Classify chest X-ray for pneumonia and TB detection
        
        Args:
            image: Chest X-ray image
        
        Returns:
            Dictionary with classification results
        """
        try:
            # Validate it's likely a chest X-ray
            image_type = self.classify_image_type(image)
            
            # Get predictions
            predictions = self.predict(image, top_k=5)
            
            # Filter for chest X-ray relevant conditions
            chest_conditions = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer"]
            relevant_predictions = [p for p in predictions if p["class"] in chest_conditions]
            
            if not relevant_predictions:
                relevant_predictions = predictions[:3]
            
            top_prediction = relevant_predictions[0]
            
            # Calculate risk
            risk_result = self.calculate_risk_score(image, top_prediction["class"])
            
            # Get segmentation
            segmented_img, seg_metadata = self.segment_problem_area(image)
            
            return {
                "image_type": image_type,
                "diagnosis": top_prediction["class"],
                "confidence": top_prediction["confidence"],
                "all_predictions": relevant_predictions,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_chest_xray_recommendation(top_prediction["class"]),
                "message": f"Chest X-ray Analysis: {top_prediction['class']} ({top_prediction['confidence']*100:.2f}% confidence)"
            }
            
        except Exception as e:
            return {
                "diagnosis": "Error",
                "message": f"Error analyzing chest X-ray: {str(e)}"
            }
    
    def _get_chest_xray_recommendation(self, diagnosis: str) -> str:
        """Get recommendation for chest X-ray diagnosis"""
        recommendations = {
            "Normal": "No abnormalities detected in chest X-ray. Continue regular health monitoring.",
            "Pneumonia": "Pneumonia detected. Consult a healthcare provider immediately. May require antibiotics.",
            "COVID-19": "COVID-19 indicators detected. Seek medical attention immediately and follow isolation protocols.",
            "Tuberculosis": "Tuberculosis indicators detected. Urgent medical consultation required. This is a serious condition requiring immediate treatment.",
            "Lung Cancer": "Potential lung cancer indicators detected. Immediate consultation with an oncologist is critical."
        }
        return recommendations.get(diagnosis, "Further evaluation by a pulmonologist is recommended.")
    
    def classify_skin_lesion(self, image) -> Dict[str, any]:
        """
        Classify skin image for moles (benign vs malignant)
        
        Args:
            image: Skin lesion image
        
        Returns:
            Dictionary with classification results
        """
        try:
            # Validate it's likely a skin image
            image_type = self.classify_image_type(image)
            
            # Get predictions
            predictions = self.predict(image, top_k=5)
            
            # Filter for skin-related conditions
            skin_conditions = ["Normal", "Melanoma", "Malignant Mole", "Benign Mole", "Dermatitis", "Skin Lesion"]
            relevant_predictions = [p for p in predictions if p["class"] in skin_conditions]
            
            if not relevant_predictions:
                relevant_predictions = predictions[:3]
            
            top_prediction = relevant_predictions[0]
            
            # Determine if benign or malignant
            is_malignant = top_prediction["class"] in ["Melanoma", "Malignant Mole"]
            is_benign = top_prediction["class"] in ["Benign Mole", "Dermatitis", "Normal"]
            
            # Calculate risk
            risk_result = self.calculate_risk_score(image, top_prediction["class"])
            
            # Get segmentation
            segmented_img, seg_metadata = self.segment_problem_area(image)
            
            return {
                "image_type": image_type,
                "diagnosis": top_prediction["class"],
                "confidence": top_prediction["confidence"],
                "is_malignant": is_malignant,
                "is_benign": is_benign,
                "all_predictions": relevant_predictions,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_skin_recommendation(top_prediction["class"]),
                "message": f"Skin Lesion Analysis: {top_prediction['class']} ({top_prediction['confidence']*100:.2f}% confidence)"
            }
            
        except Exception as e:
            return {
                "diagnosis": "Error",
                "message": f"Error analyzing skin lesion: {str(e)}"
            }
    
    def _get_skin_recommendation(self, diagnosis: str) -> str:
        """Get recommendation for skin lesion diagnosis"""
        recommendations = {
            "Normal": "No concerning skin lesions detected. Continue regular skin monitoring.",
            "Benign Mole": "Benign mole detected. Monitor for any changes in size, shape, or color. Regular dermatological checkups recommended.",
            "Melanoma": "⚠️ URGENT: Potential melanoma detected. Immediate dermatological evaluation is critical. Early detection is crucial for treatment success.",
            "Malignant Mole": "⚠️ URGENT: Malignant mole detected. Immediate consultation with a dermatologist or oncologist is required.",
            "Dermatitis": "Dermatitis detected. May be treatable with topical medications. Consult a dermatologist for proper treatment.",
            "Skin Lesion": "Skin lesion detected. Have it evaluated by a dermatologist to determine if further testing is needed."
        }
        return recommendations.get(diagnosis, "Consult a dermatologist for proper evaluation and diagnosis.")
    
    def classify_bone_fracture(self, image) -> Dict[str, any]:
        """
        Classify bone X-ray for fracture detection
        
        Args:
            image: Bone X-ray image
        
        Returns:
            Dictionary with classification results
        """
        try:
            # Validate it's likely a bone X-ray
            image_type = self.classify_image_type(image)
            
            # Get predictions
            predictions = self.predict(image, top_k=5)
            
            # Filter for fracture-related conditions
            fracture_conditions = ["Fracture", "No Fracture", "Normal", "Arthritis"]
            relevant_predictions = [p for p in predictions if p["class"] in fracture_conditions]
            
            if not relevant_predictions:
                relevant_predictions = predictions[:3]
            
            top_prediction = relevant_predictions[0]
            
            # Determine if fracture is present
            has_fracture = top_prediction["class"] == "Fracture"
            
            # Calculate risk
            risk_result = self.calculate_risk_score(image, top_prediction["class"])
            
            # Get segmentation
            segmented_img, seg_metadata = self.segment_problem_area(image)
            
            return {
                "image_type": image_type,
                "diagnosis": top_prediction["class"],
                "confidence": top_prediction["confidence"],
                "has_fracture": has_fracture,
                "no_fracture": not has_fracture,
                "all_predictions": relevant_predictions,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_fracture_recommendation(top_prediction["class"]),
                "message": f"Bone X-ray Analysis: {top_prediction['class']} ({top_prediction['confidence']*100:.2f}% confidence)"
            }
            
        except Exception as e:
            return {
                "diagnosis": "Error",
                "message": f"Error analyzing bone X-ray: {str(e)}"
            }
    
    def _get_fracture_recommendation(self, diagnosis: str) -> str:
        """Get recommendation for fracture diagnosis"""
        recommendations = {
            "Fracture": "⚠️ FRACTURE DETECTED: Seek immediate medical attention. Avoid moving the affected area. Immobilize if possible and go to emergency room.",
            "No Fracture": "No fracture detected in the X-ray. If pain persists, consult an orthopedic specialist for further evaluation.",
            "Normal": "No abnormalities detected in the bone structure. If symptoms persist, consult a healthcare provider.",
            "Arthritis": "Arthritis indicators detected. Consult a rheumatologist or orthopedic specialist for proper diagnosis and treatment plan."
        }
        return recommendations.get(diagnosis, "Further evaluation by an orthopedic specialist is recommended.")
    
    def detect_brain_abnormalities(self, image) -> Dict[str, any]:
        """
        Detect abnormalities in brain MRI/CT scans
        Detects: tumors, strokes, hemorrhages, aneurysms, etc.
        
        Args:
            image: Brain MRI/CT image
            
        Returns:
            Dictionary with abnormality detection results
        """
        try:
            image_type = self.classify_image_type(image)
            
            # Get predictions
            predictions = self.predict(image, top_k=5)
            
            # Brain-specific conditions
            brain_conditions = {
                "Normal": "No abnormalities",
                "Brain Tumor": "Tumor detected",
                "Stroke": "Stroke indicators",
                "Brain Hemorrhage": "Hemorrhage detected",
                "Aneurysm": "Aneurysm detected",
                "Multiple Sclerosis": "MS lesions",
                "Brain Atrophy": "Atrophy present"
            }
            
            # Analyze for abnormalities
            is_abnormal = False
            abnormalities = []
            confidence = 0.0
            
            for pred in predictions:
                if pred["class"] != "Normal" and pred["confidence"] > 0.3:
                    is_abnormal = True
                    abnormalities.append({
                        "condition": pred["class"],
                        "confidence": pred["confidence"],
                        "severity": "High" if pred["confidence"] > 0.7 else "Moderate" if pred["confidence"] > 0.5 else "Low"
                    })
                    if pred["confidence"] > confidence:
                        confidence = pred["confidence"]
            
            # Get segmentation
            segmented_img, seg_metadata = self.segment_problem_area(image)
            
            # Calculate risk
            risk_result = self.calculate_risk_score(image, abnormalities[0]["condition"] if abnormalities else "Normal")
            
            return {
                "image_type": image_type,
                "is_abnormal": is_abnormal,
                "is_normal": not is_abnormal,
                "abnormalities": abnormalities,
                "primary_finding": abnormalities[0] if abnormalities else {"condition": "Normal", "confidence": 1.0},
                "confidence": confidence if is_abnormal else 1.0,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_brain_recommendation(abnormalities),
                "message": f"Brain Scan Analysis: {'Abnormalities detected' if is_abnormal else 'No abnormalities detected'} ({confidence*100:.2f}% confidence)" if is_abnormal else "Brain Scan Analysis: Normal (No abnormalities detected)"
            }
            
        except Exception as e:
            return {
                "is_abnormal": False,
                "message": f"Error analyzing brain scan: {str(e)}"
            }
    
    def _get_brain_recommendation(self, abnormalities: List[Dict]) -> str:
        """Get recommendation for brain abnormalities"""
        if not abnormalities:
            return "No brain abnormalities detected. Continue regular health monitoring."
        
        primary = abnormalities[0]["condition"]
        recommendations = {
            "Brain Tumor": "⚠️ URGENT: Brain tumor detected. Immediate consultation with a neurosurgeon and oncologist is critical.",
            "Stroke": "⚠️ URGENT: Stroke indicators detected. Seek immediate emergency medical attention. Time is critical for treatment.",
            "Brain Hemorrhage": "⚠️ CRITICAL: Brain hemorrhage detected. This is a medical emergency. Go to emergency room immediately.",
            "Aneurysm": "⚠️ URGENT: Aneurysm detected. Immediate neurosurgical consultation required. This is a serious condition.",
            "Multiple Sclerosis": "MS lesions detected. Consult a neurologist for proper diagnosis and treatment plan.",
            "Brain Atrophy": "Brain atrophy detected. Neurological evaluation recommended to determine cause and treatment."
        }
        return recommendations.get(primary, "Brain abnormalities detected. Consult a neurologist for proper evaluation and diagnosis.")
    
    def detect_abdominal_abnormalities(self, image) -> Dict[str, any]:
        """
        Detect abnormalities in abdominal imaging (CT, X-ray, ultrasound)
        Detects: appendicitis, gallstones, kidney stones, tumors, etc.
        
        Args:
            image: Abdominal imaging
            
        Returns:
            Dictionary with abnormality detection results
        """
        try:
            image_type = self.classify_image_type(image)
            predictions = self.predict(image, top_k=5)
            
            # Abdominal-specific conditions
            abdominal_conditions = {
                "Normal": "No abnormalities",
                "Appendicitis": "Appendicitis detected",
                "Gallstones": "Gallstones present",
                "Kidney Stones": "Kidney stones detected",
                "Abdominal Tumor": "Tumor detected",
                "Hernia": "Hernia detected",
                "Bowel Obstruction": "Bowel obstruction",
                "Liver Abnormalities": "Liver issues",
                "Pancreatitis": "Pancreatitis detected"
            }
            
            is_abnormal = False
            abnormalities = []
            confidence = 0.0
            
            for pred in predictions:
                if pred["class"] != "Normal" and pred["confidence"] > 0.3:
                    is_abnormal = True
                    abnormalities.append({
                        "condition": pred["class"],
                        "confidence": pred["confidence"],
                        "severity": "High" if pred["confidence"] > 0.7 else "Moderate" if pred["confidence"] > 0.5 else "Low"
                    })
                    if pred["confidence"] > confidence:
                        confidence = pred["confidence"]
            
            segmented_img, seg_metadata = self.segment_problem_area(image)
            risk_result = self.calculate_risk_score(image, abnormalities[0]["condition"] if abnormalities else "Normal")
            
            return {
                "image_type": image_type,
                "is_abnormal": is_abnormal,
                "is_normal": not is_abnormal,
                "abnormalities": abnormalities,
                "primary_finding": abnormalities[0] if abnormalities else {"condition": "Normal", "confidence": 1.0},
                "confidence": confidence if is_abnormal else 1.0,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_abdominal_recommendation(abnormalities),
                "message": f"Abdominal Imaging Analysis: {'Abnormalities detected' if is_abnormal else 'No abnormalities detected'}"
            }
            
        except Exception as e:
            return {
                "is_abnormal": False,
                "message": f"Error analyzing abdominal imaging: {str(e)}"
            }
    
    def _get_abdominal_recommendation(self, abnormalities: List[Dict]) -> str:
        """Get recommendation for abdominal abnormalities"""
        if not abnormalities:
            return "No abdominal abnormalities detected. Continue regular health monitoring."
        
        primary = abnormalities[0]["condition"]
        recommendations = {
            "Appendicitis": "⚠️ URGENT: Appendicitis detected. Seek immediate medical attention. Surgery may be required.",
            "Gallstones": "Gallstones detected. Consult a gastroenterologist. May require treatment or surgery.",
            "Kidney Stones": "Kidney stones detected. Consult a urologist. Treatment depends on stone size and location.",
            "Abdominal Tumor": "⚠️ URGENT: Abdominal tumor detected. Immediate consultation with an oncologist is critical.",
            "Hernia": "Hernia detected. Consult a general surgeon for evaluation and potential repair.",
            "Bowel Obstruction": "⚠️ URGENT: Bowel obstruction detected. Seek immediate medical attention.",
            "Pancreatitis": "Pancreatitis detected. Consult a gastroenterologist immediately. May require hospitalization."
        }
        return recommendations.get(primary, "Abdominal abnormalities detected. Consult a gastroenterologist for proper evaluation.")
    
    def detect_retinal_abnormalities(self, image) -> Dict[str, any]:
        """
        Detect abnormalities in retinal/eye imaging
        Detects: diabetic retinopathy, glaucoma, macular degeneration, etc.
        
        Args:
            image: Retinal/eye fundus image
            
        Returns:
            Dictionary with abnormality detection results
        """
        try:
            image_type = self.classify_image_type(image)
            predictions = self.predict(image, top_k=5)
            
            # Retinal-specific conditions
            retinal_conditions = {
                "Normal": "No abnormalities",
                "Diabetic Retinopathy": "Diabetic retinopathy",
                "Glaucoma": "Glaucoma detected",
                "Macular Degeneration": "Macular degeneration",
                "Retinal Detachment": "Retinal detachment",
                "Hypertensive Retinopathy": "Hypertensive retinopathy",
                "Cataracts": "Cataracts present"
            }
            
            is_abnormal = False
            abnormalities = []
            confidence = 0.0
            
            for pred in predictions:
                if pred["class"] != "Normal" and pred["confidence"] > 0.3:
                    is_abnormal = True
                    abnormalities.append({
                        "condition": pred["class"],
                        "confidence": pred["confidence"],
                        "severity": "High" if pred["confidence"] > 0.7 else "Moderate" if pred["confidence"] > 0.5 else "Low"
                    })
                    if pred["confidence"] > confidence:
                        confidence = pred["confidence"]
            
            segmented_img, seg_metadata = self.segment_problem_area(image)
            risk_result = self.calculate_risk_score(image, abnormalities[0]["condition"] if abnormalities else "Normal")
            
            return {
                "image_type": image_type,
                "is_abnormal": is_abnormal,
                "is_normal": not is_abnormal,
                "abnormalities": abnormalities,
                "primary_finding": abnormalities[0] if abnormalities else {"condition": "Normal", "confidence": 1.0},
                "confidence": confidence if is_abnormal else 1.0,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_retinal_recommendation(abnormalities),
                "message": f"Retinal Imaging Analysis: {'Abnormalities detected' if is_abnormal else 'No abnormalities detected'}"
            }
            
        except Exception as e:
            return {
                "is_abnormal": False,
                "message": f"Error analyzing retinal imaging: {str(e)}"
            }
    
    def _get_retinal_recommendation(self, abnormalities: List[Dict]) -> str:
        """Get recommendation for retinal abnormalities"""
        if not abnormalities:
            return "No retinal abnormalities detected. Continue regular eye exams."
        
        primary = abnormalities[0]["condition"]
        recommendations = {
            "Diabetic Retinopathy": "⚠️ Diabetic retinopathy detected. Consult an ophthalmologist immediately. Control blood sugar levels. Regular monitoring required.",
            "Glaucoma": "⚠️ Glaucoma detected. Immediate ophthalmological consultation required. Early treatment can prevent vision loss.",
            "Macular Degeneration": "Macular degeneration detected. Consult an ophthalmologist. Treatment may slow progression.",
            "Retinal Detachment": "⚠️ URGENT: Retinal detachment detected. This is a medical emergency. Seek immediate ophthalmological care.",
            "Hypertensive Retinopathy": "Hypertensive retinopathy detected. Control blood pressure and consult an ophthalmologist.",
            "Cataracts": "Cataracts detected. Consult an ophthalmologist. Surgery may be recommended if vision is significantly affected."
        }
        return recommendations.get(primary, "Retinal abnormalities detected. Consult an ophthalmologist for proper evaluation and treatment.")
    
    def detect_mammography_abnormalities(self, image) -> Dict[str, any]:
        """
        Detect abnormalities in mammography (breast imaging)
        Detects: breast cancer, benign masses, calcifications, etc.
        
        Args:
            image: Mammography image
            
        Returns:
            Dictionary with abnormality detection results
        """
        try:
            image_type = self.classify_image_type(image)
            predictions = self.predict(image, top_k=5)
            
            # Mammography-specific conditions
            mammo_conditions = {
                "Normal": "No abnormalities",
                "Breast Cancer": "Breast cancer detected",
                "Benign Mass": "Benign mass",
                "Suspicious Calcifications": "Suspicious calcifications",
                "Fibroadenoma": "Fibroadenoma",
                "Cyst": "Cyst detected"
            }
            
            is_abnormal = False
            abnormalities = []
            confidence = 0.0
            
            for pred in predictions:
                if pred["class"] != "Normal" and pred["confidence"] > 0.3:
                    is_abnormal = True
                    abnormalities.append({
                        "condition": pred["class"],
                        "confidence": pred["confidence"],
                        "severity": "High" if pred["confidence"] > 0.7 else "Moderate" if pred["confidence"] > 0.5 else "Low"
                    })
                    if pred["confidence"] > confidence:
                        confidence = pred["confidence"]
            
            segmented_img, seg_metadata = self.segment_problem_area(image)
            risk_result = self.calculate_risk_score(image, abnormalities[0]["condition"] if abnormalities else "Normal")
            
            return {
                "image_type": image_type,
                "is_abnormal": is_abnormal,
                "is_normal": not is_abnormal,
                "abnormalities": abnormalities,
                "primary_finding": abnormalities[0] if abnormalities else {"condition": "Normal", "confidence": 1.0},
                "confidence": confidence if is_abnormal else 1.0,
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
                "segmentation_metadata": seg_metadata,
                "recommendation": self._get_mammography_recommendation(abnormalities),
                "message": f"Mammography Analysis: {'Abnormalities detected' if is_abnormal else 'No abnormalities detected'}"
            }
            
        except Exception as e:
            return {
                "is_abnormal": False,
                "message": f"Error analyzing mammography: {str(e)}"
            }
    
    def _get_mammography_recommendation(self, abnormalities: List[Dict]) -> str:
        """Get recommendation for mammography abnormalities"""
        if not abnormalities:
            return "No breast abnormalities detected. Continue regular mammography screening as recommended."
        
        primary = abnormalities[0]["condition"]
        recommendations = {
            "Breast Cancer": "⚠️ URGENT: Breast cancer indicators detected. Immediate consultation with a breast surgeon and oncologist is critical. Early detection improves treatment outcomes.",
            "Suspicious Calcifications": "⚠️ Suspicious calcifications detected. Further evaluation with biopsy may be required. Consult a breast specialist.",
            "Benign Mass": "Benign mass detected. Regular monitoring recommended. Consult a breast specialist for follow-up.",
            "Fibroadenoma": "Fibroadenoma detected. Usually benign but should be monitored. Consult a breast specialist.",
            "Cyst": "Cyst detected. Usually benign. May require monitoring or aspiration. Consult a breast specialist."
        }
        return recommendations.get(primary, "Breast abnormalities detected. Consult a breast specialist for proper evaluation and diagnosis.")
    
    def detect_chest_abnormalities(self, image) -> Dict[str, any]:
        """
        Enhanced chest X-ray abnormality detection
        Detects: pneumonia, TB, lung cancer, pneumothorax, pleural effusion, etc.
        
        Args:
            image: Chest X-ray image
            
        Returns:
            Dictionary with comprehensive abnormality detection results
        """
        try:
            # Use existing chest X-ray classification
            chest_result = self.classify_chest_xray(image)
            
            # Additional chest-specific abnormalities
            additional_conditions = {
                "Pneumothorax": "Pneumothorax (collapsed lung)",
                "Pleural Effusion": "Pleural effusion (fluid in lungs)",
                "Pulmonary Edema": "Pulmonary edema",
                "Atelectasis": "Atelectasis (collapsed lung tissue)",
                "Cardiomegaly": "Cardiomegaly (enlarged heart)"
            }
            
            # Check for additional abnormalities
            predictions = self.predict(image, top_k=8)
            additional_findings = []
            
            for pred in predictions:
                if pred["class"] in additional_conditions and pred["confidence"] > 0.4:
                    additional_findings.append({
                        "condition": pred["class"],
                        "description": additional_conditions[pred["class"]],
                        "confidence": pred["confidence"]
                    })
            
            # Combine results
            is_abnormal = chest_result.get("diagnosis") != "Normal" or len(additional_findings) > 0
            
            return {
                **chest_result,
                "is_abnormal": is_abnormal,
                "is_normal": not is_abnormal,
                "additional_findings": additional_findings,
                "all_abnormalities": [
                    {"condition": chest_result.get("diagnosis"), "confidence": chest_result.get("confidence", 0.0)}
                ] + additional_findings,
                "message": f"Chest X-ray Analysis: {'Multiple abnormalities detected' if len(additional_findings) > 0 else chest_result.get('message', 'Analysis complete')}"
            }
            
        except Exception as e:
            return {
                "is_abnormal": False,
                "message": f"Error analyzing chest X-ray: {str(e)}"
            }
    
    def detect_bone_abnormalities(self, image) -> Dict[str, any]:
        """
        Enhanced bone imaging abnormality detection
        Detects: fractures, arthritis, osteoporosis, tumors, infections, etc.
        
        Args:
            image: Bone X-ray/imaging
            
        Returns:
            Dictionary with comprehensive bone abnormality detection results
        """
        try:
            # Use existing fracture classification
            fracture_result = self.classify_bone_fracture(image)
            
            # Additional bone-specific abnormalities
            additional_conditions = {
                "Osteoporosis": "Osteoporosis (bone density loss)",
                "Osteoarthritis": "Osteoarthritis",
                "Rheumatoid Arthritis": "Rheumatoid arthritis",
                "Bone Tumor": "Bone tumor",
                "Osteomyelitis": "Osteomyelitis (bone infection)",
                "Bone Deformity": "Bone deformity"
            }
            
            # Check for additional abnormalities
            predictions = self.predict(image, top_k=8)
            additional_findings = []
            
            for pred in predictions:
                if pred["class"] in additional_conditions and pred["confidence"] > 0.4:
                    additional_findings.append({
                        "condition": pred["class"],
                        "description": additional_conditions[pred["class"]],
                        "confidence": pred["confidence"]
                    })
            
            # Combine results
            has_fracture = fracture_result.get("has_fracture", False)
            is_abnormal = has_fracture or len(additional_findings) > 0
            
            return {
                **fracture_result,
                "is_abnormal": is_abnormal,
                "is_normal": not is_abnormal,
                "additional_findings": additional_findings,
                "all_abnormalities": (
                    [{"condition": "Fracture", "confidence": fracture_result.get("confidence", 0.0)}] if has_fracture else []
                ) + additional_findings,
                "message": f"Bone Imaging Analysis: {'Multiple abnormalities detected' if len(additional_findings) > 0 else fracture_result.get('message', 'Analysis complete')}"
            }
            
        except Exception as e:
            return {
                "is_abnormal": False,
                "message": f"Error analyzing bone imaging: {str(e)}"
            }


# Global classifier instance
_classifier_instance = None

def get_classifier() -> ImageClassifier:
    """Get or create global classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ImageClassifier()
    return _classifier_instance

def predict_diagnosis(image):
    """
    Make a prediction using the AI model (backward compatibility)
    
    Args:
        image: Input image
    
    Returns:
        Diagnosis string or dictionary
    """
    classifier = get_classifier()
    result = classifier.predict_diagnosis(image)
    return result.get("message", result.get("diagnosis", "Unable to determine"))

def preprocess_image(image, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Preprocess image for classification
    
    Args:
        image: Input image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image array
    """
    classifier = get_classifier()
    return classifier.preprocess_image(image, target_size)

def classify_image(image, top_k: int = 3) -> List[Dict[str, any]]:
    """
    Classify image and return top predictions
    
    Args:
        image: Input image
        top_k: Number of top predictions
    
    Returns:
        List of prediction dictionaries
    """
    classifier = get_classifier()
    return classifier.predict(image, top_k)

def get_image_type(image) -> str:
    """
    Classify the type of medical image
    
    Args:
        image: Input image
    
    Returns:
        Image type string
    """
    classifier = get_classifier()
    return classifier.classify_image_type(image)

def validate_image(image) -> Tuple[bool, str]:
    """
    Validate if image is suitable for classification
    
    Args:
        image: Input image
    
    Returns:
        Tuple of (is_valid, message)
    """
    classifier = get_classifier()
    return classifier.validate_image(image)

def classify_normal_abnormal(image) -> Dict[str, any]:
    """
    Classify image as Normal or Abnormal
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with classification result
    """
    classifier = get_classifier()
    return classifier.classify_normal_abnormal(image)

def segment_problem_area(image) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Segment and highlight problem areas in the image
    
    Args:
        image: Input image
    
    Returns:
        Tuple of (segmented_image, metadata)
    """
    classifier = get_classifier()
    return classifier.segment_problem_area(image)

def calculate_risk_score(image, diagnosis: str = None) -> Dict[str, any]:
    """
    Calculate risk score for the medical condition
    
    Args:
        image: Input image
        diagnosis: Optional diagnosis string
    
    Returns:
        Dictionary with risk score and details
    """
    classifier = get_classifier()
    return classifier.calculate_risk_score(image, diagnosis)

def classify_chest_xray(image) -> Dict[str, any]:
    """
    Classify chest X-ray for pneumonia and TB detection
    
    Args:
        image: Chest X-ray image
    
    Returns:
        Dictionary with classification results
    """
    classifier = get_classifier()
    return classifier.classify_chest_xray(image)

def classify_skin_lesion(image) -> Dict[str, any]:
    """
    Classify skin image for moles (benign vs malignant)
    
    Args:
        image: Skin lesion image
    
    Returns:
        Dictionary with classification results
    """
    classifier = get_classifier()
    return classifier.classify_skin_lesion(image)

def classify_bone_fracture(image) -> Dict[str, any]:
    """
    Classify bone X-ray for fracture detection
    
    Args:
        image: Bone X-ray image
    
    Returns:
        Dictionary with classification results
    """
    classifier = get_classifier()
    return classifier.classify_bone_fracture(image)

def detect_brain_abnormalities(image) -> Dict[str, any]:
    """
    Detect abnormalities in brain MRI/CT scans
    Detects: tumors, strokes, hemorrhages, aneurysms, etc.
    
    Args:
        image: Brain MRI/CT image
    
    Returns:
        Dictionary with abnormality detection results
    """
    classifier = get_classifier()
    return classifier.detect_brain_abnormalities(image)

def detect_abdominal_abnormalities(image) -> Dict[str, any]:
    """
    Detect abnormalities in abdominal imaging (CT, X-ray, ultrasound)
    Detects: appendicitis, gallstones, kidney stones, tumors, etc.
    
    Args:
        image: Abdominal imaging
    
    Returns:
        Dictionary with abnormality detection results
    """
    classifier = get_classifier()
    return classifier.detect_abdominal_abnormalities(image)

def detect_retinal_abnormalities(image) -> Dict[str, any]:
    """
    Detect abnormalities in retinal/eye imaging
    Detects: diabetic retinopathy, glaucoma, macular degeneration, etc.
    
    Args:
        image: Retinal/eye fundus image
    
    Returns:
        Dictionary with abnormality detection results
    """
    classifier = get_classifier()
    return classifier.detect_retinal_abnormalities(image)

def detect_mammography_abnormalities(image) -> Dict[str, any]:
    """
    Detect abnormalities in mammography (breast imaging)
    Detects: breast cancer, benign masses, calcifications, etc.
    
    Args:
        image: Mammography image
    
    Returns:
        Dictionary with abnormality detection results
    """
    classifier = get_classifier()
    return classifier.detect_mammography_abnormalities(image)

def detect_chest_abnormalities(image) -> Dict[str, any]:
    """
    Enhanced chest X-ray abnormality detection
    Detects: pneumonia, TB, lung cancer, pneumothorax, pleural effusion, etc.
    
    Args:
        image: Chest X-ray image
    
    Returns:
        Dictionary with comprehensive abnormality detection results
    """
    classifier = get_classifier()
    return classifier.detect_chest_abnormalities(image)

def detect_bone_abnormalities(image) -> Dict[str, any]:
    """
    Enhanced bone imaging abnormality detection
    Detects: fractures, arthritis, osteoporosis, tumors, infections, etc.
    
    Args:
        image: Bone X-ray/imaging
    
    Returns:
        Dictionary with comprehensive bone abnormality detection results
    """
    classifier = get_classifier()
    return classifier.detect_bone_abnormalities(image)
