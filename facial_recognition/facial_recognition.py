'''
Docstring for facial_recognition.facial_recognition
Facial recognition using InsightFace with ONNX Runtime.
Implementation with YOLO bounding boxes.
'''

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from typing import List, Tuple, Dict, Optional, Any
import os

class FacialRecognition:
    """
    Facial recognition using InsightFace with ONNX runtime.
    Designed to work with YOLO detection boxes.
    """
    
    def __init__(self, 
                 model_name: str = "buffalo_l",
                 providers: List[str] = None,
                 det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize InsightFace with ONNX runtime.
        
        Args:
            model_name: InsightFace model pack name ('buffalo_l', 'buffalo_s', 'buffalo_sc')
            providers: ONNX execution providers ['CUDAExecutionProvider', 'CPUExecutionProvider']
            det_size: Detection input size
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Initialize FaceAnalysis with ONNX backend
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

        # Store known face embeddings: {person_id: embedding}
        self.known_faces: Dict[str, np.ndarray] = {}

    def extract_face_from_yolo_box(self,
                                   image: np.ndarray,
                                   yolo_box: Tuple[int, int, int, int],
                                   padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face region from YOLO person/face bounding box.
        
        Args:
            image: Full image (BGR format)
            yolo_box: YOLO box coordinates (x1, y1, x2, y2)
            padding: Padding ratio to add around the box
            
        Returns:
            Cropped face region or None
        """
        x1, y1, x2, y2 = yolo_box
        h, w = image.shape[:2]
        
        # Add padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return image[y1:y2, x1:x2]

    def get_faces_from_crop(self, face_crop: np.ndarray) -> List[Any]:
        """
        Get all detected faces from a cropped image.
        
        Args:
            face_crop: Cropped face image (BGR format)
            
        Returns:
            List of InsightFace face objects with all attributes
        """
        if face_crop is None or face_crop.size == 0:
            return []
        return self.app.get(face_crop)

    def get_faces_from_yolo_detection(self,
                                      image: np.ndarray,
                                      yolo_box: Tuple[int, int, int, int],
                                      padding: float = 0.2) -> List[Any]:
        """
        Get all faces from a YOLO detection box.
        
        Args:
            image: Full image (BGR format)
            yolo_box: YOLO coordinates (x1, y1, x2, y2)
            padding: Padding ratio
            
        Returns:
            List of InsightFace face objects
        """
        crop = self.extract_face_from_yolo_box(image, yolo_box, padding)
        return self.get_faces_from_crop(crop)

    def get_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from a cropped image using InsightFace.
        
        Args:
            face_crop: Cropped face image (BGR format)
            
        Returns:
            512-dimensional face embedding or None if no face detected
        """
        faces = self.get_faces_from_crop(face_crop)
        
        if len(faces) == 0:
            return None
        
        # Return embedding of the largest/most confident face
        face = max(faces, key=lambda x: x.det_score)
        return face.normed_embedding
    
    def get_embedding_from_yolo_detection(self,
                                           image: np.ndarray,
                                           yolo_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get face embedding directly from YOLO detection box.
        
        Args:
            image: Full image (BGR format)
            yolo_box: YOLO coordinates (x1, y1, x2, y2)
            
        Returns:
            Face embedding or None
        """
        crop = self.extract_face_from_yolo_box(image, yolo_box)
        if crop is None or crop.size == 0:
            return None
        return self.get_embedding_from_crop(crop)
    
    def register_face(self, person_id: str, embedding: np.ndarray):
        """Register a face embedding with a person ID."""
        self.known_faces[person_id] = embedding
        print(f"Registered face for: {person_id}")