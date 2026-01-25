import cv2
import numpy as np
import streamlit as st
from PIL import Image
from medmnist import ChestMNIST, ChestXRayMNIST, ChestXRaySmallMNIST, ChestXRayLargeMNIST

def apply_infrared_effect(image):
    """
    Create a more realistic infrared effect
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply false coloring (hot objects appear brighter)
        infrared = cv2.applyColorMap(equalized, cv2.COLORMAP_HOT)
        
        # Enhance contrast
        alpha = 1.2  # Increase for more contrast
        beta = 10    # Increase for more brightness
        infrared = cv2.convertScaleAbs(infrared, alpha=alpha, beta=beta)
        
        return infrared
    except Exception as e:
        print(f"Error applying infrared effect: {e}")
        return image

def apply_filter(image, filter_type):
    """
    Apply different filters to the image
    """
    try:
        if filter_type == "Grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif filter_type == "Infrared":
            return apply_infrared_effect(image)
        elif filter_type == "Xray":
            # Simulate X-ray effect
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            return cv2.equalizeHist(inverted)
        else:
            return enhance_image_clarity(image)
    except Exception as e:
        print(f"Error applying filter: {e}")
        return image

def enhance_image_clarity(image):
    """
    Enhance image clarity using multiple techniques
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge channels
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)

        # Reduce noise
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

        return denoised

    except Exception as e:
        print(f"Error enhancing image clarity: {e}")
        return image

def process_image(image, filter_type="Original"):
    """
    Process the uploaded or captured image
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
        
        # Apply selected filter
        filtered_image = apply_filter(image, filter_type)
        
        return filtered_image
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
