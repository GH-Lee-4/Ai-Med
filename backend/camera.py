import cv2

def capture_image():
    """
    Captures an image from the webcam
    """
    try:
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return None
            
        # Capture frame
        ret, frame = cap.read()
        
        # Release the webcam
        cap.release()
        
        if ret:
            return frame
        return None
        
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None
