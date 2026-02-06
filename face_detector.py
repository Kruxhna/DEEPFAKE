"""
Face Detection Module for Deepfake Detection
Uses MTCNN for robust face detection and extraction
"""

import cv2
import numpy as np
import torch
from PIL import Image

# Try to import MTCNN, fall back to OpenCV if not available
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Using OpenCV face detection.")


class FaceDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks)
    Falls back to OpenCV Haar Cascade if MTCNN is not available
    """
    
    def __init__(self, device=None, min_face_size=60, margin=40):
        """
        Initialize face detector
        
        Args:
            device: torch device (cuda/cpu)
            min_face_size: Minimum face size to detect (pixels)
            margin: Margin to add around detected face (pixels)
        """
        self.margin = margin
        self.min_face_size = min_face_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=224,
                margin=margin,
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
                factor=0.709,  # Scale factor
                post_process=False,
                device=self.device,
                keep_all=True  # Return all faces
            )
            self.detection_method = "MTCNN"
        else:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.detection_method = "OpenCV"
        
        print(f"[FaceDetector] Using {self.detection_method} on {self.device}")
    
    def detect_faces(self, image):
        """
        Detect all faces in an image
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
            Empty list if no faces found
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
            image_rgb = np.array(pil_image)
        
        boxes = []
        
        if MTCNN_AVAILABLE:
            # MTCNN detection
            detected_boxes, probs = self.detector.detect(pil_image)
            
            if detected_boxes is not None:
                for box, prob in zip(detected_boxes, probs):
                    if prob > 0.9:  # High confidence threshold
                        x1, y1, x2, y2 = [int(b) for b in box]
                        boxes.append((x1, y1, x2, y2))
        else:
            # OpenCV Haar Cascade detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            for (x, y, w, h) in faces:
                boxes.append((x, y, x + w, y + h))
        
        return boxes
    
    def extract_face(self, image, box=None, target_size=(224, 224)):
        """
        Extract and resize a single face from image
        
        Args:
            image: numpy array (BGR) or PIL Image
            box: Optional bounding box (x1, y1, x2, y2). If None, detect largest face.
            target_size: Output size (width, height)
            
        Returns:
            Face crop as numpy array (RGB), or None if no face found
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            # PIL is RGB, keep as is
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        if box is None:
            # Detect faces and use the largest one
            boxes = self.detect_faces(image)
            if not boxes:
                return None
            
            # Find largest face by area
            largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            box = largest_box
        
        x1, y1, x2, y2 = box
        
        # Add margin
        margin = self.margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size)
        
        return face_resized
    
    def extract_all_faces(self, image, target_size=(224, 224)):
        """
        Extract all faces from an image
        
        Args:
            image: numpy array (BGR) or PIL Image
            target_size: Output size (width, height)
            
        Returns:
            List of face crops as numpy arrays (RGB)
        """
        boxes = self.detect_faces(image)
        faces = []
        
        for box in boxes:
            face = self.extract_face(image, box, target_size)
            if face is not None:
                faces.append(face)
        
        return faces
    
    def extract_face_or_frame(self, image, target_size=(224, 224)):
        """
        Extract face if found, otherwise return resized full frame
        
        Args:
            image: numpy array (BGR) or PIL Image
            target_size: Output size (width, height)
            
        Returns:
            Face crop or resized frame as numpy array (RGB)
        """
        face = self.extract_face(image, target_size=target_size)
        
        if face is not None:
            return face, True  # Return face and flag indicating face was found
        
        # No face found, return resized frame
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(image, target_size)
        return resized, False  # Return frame and flag indicating no face found


# Singleton instance for convenience
_default_detector = None

def get_face_detector(device=None):
    """Get or create default face detector instance"""
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector(device=device)
    return _default_detector


def extract_face_from_frame(frame, target_size=(224, 224), detector=None):
    """
    Convenience function to extract face from a frame
    
    Args:
        frame: numpy array (BGR from OpenCV)
        target_size: Output size
        detector: Optional FaceDetector instance
        
    Returns:
        Face crop or resized frame, and boolean indicating if face was found
    """
    if detector is None:
        detector = get_face_detector()
    
    return detector.extract_face_or_frame(frame, target_size)


# Test function
if __name__ == "__main__":
    print("Testing Face Detector...")
    
    # Create detector
    detector = FaceDetector()
    
    # Create a test image (simple colored rectangle)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = (100, 150, 200)  # BGR color
    
    # Test detection (won't find real faces in synthetic image)
    boxes = detector.detect_faces(test_image)
    print(f"Detected {len(boxes)} faces in test image")
    
    # Test extract_face_or_frame
    result, face_found = detector.extract_face_or_frame(test_image)
    print(f"Face found: {face_found}, Result shape: {result.shape}")
    
    print("[OK] Face Detector module working!")
