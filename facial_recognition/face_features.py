'''
Docstring for facial_recognition.face_features
Module for extracting and analyzing facial features from InsightFace face objects.
'''

import cv2
import numpy as np
from typing import Dict, Optional, Any, List


class FaceFeatures:
    """
    Extract and analyze facial features from InsightFace face objects.
    """
    
    @staticmethod
    def extract_features(face: Any) -> Dict:
        """
        Extract all available features from an InsightFace face object.
        
        Args:
            face: InsightFace face object from FaceAnalysis.get()
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
            'det_score': float(face.det_score) if hasattr(face, 'det_score') else None,
            'embedding': face.normed_embedding if hasattr(face, 'normed_embedding') else None,
        }
        
        # 5-point landmarks (eyes, nose, mouth corners)
        if hasattr(face, 'kps') and face.kps is not None:
            features['landmarks_5'] = face.kps.tolist()
        else:
            features['landmarks_5'] = None
            
        # Age estimation
        if hasattr(face, 'age') and face.age is not None:
            features['age'] = int(face.age)
        else:
            features['age'] = None
            
        # Gender classification
        if hasattr(face, 'gender') and face.gender is not None:
            features['gender'] = 'male' if face.gender == 1 else 'female'
            features['gender_raw'] = int(face.gender)
        else:
            features['gender'] = None
            features['gender_raw'] = None
            
        # Head pose (pitch, yaw, roll)
        if hasattr(face, 'pose') and face.pose is not None:
            features['pose'] = {
                'pitch': float(face.pose[0]),
                'yaw': float(face.pose[1]),
                'roll': float(face.pose[2])
            }
        else:
            features['pose'] = None
            
        # 106-point landmarks
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            features['landmarks_106'] = face.landmark_2d_106.tolist()
        else:
            features['landmarks_106'] = None
            
        # 68-point 3D landmarks
        if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            features['landmarks_3d_68'] = face.landmark_3d_68.tolist()
        else:
            features['landmarks_3d_68'] = None
            
        return features
    
    @staticmethod
    def draw_landmarks(image: np.ndarray, 
                       features: Dict, 
                       draw_5pt: bool = True,
                       draw_106pt: bool = False) -> np.ndarray:
        """
        Draw facial landmarks on an image.
        
        Args:
            image: Image to draw on (BGR format)
            features: Features dict from extract_features()
            draw_5pt: Draw 5-point landmarks
            draw_106pt: Draw 106-point landmarks
            
        Returns:
            Image with landmarks drawn
        """
        img = image.copy()
        
        # Draw 5-point landmarks
        if draw_5pt and features.get('landmarks_5'):
            colors = [
                (0, 255, 0),    # Left eye - green
                (0, 255, 0),    # Right eye - green
                (255, 0, 0),    # Nose - blue
                (0, 0, 255),    # Left mouth - red
                (0, 0, 255),    # Right mouth - red
            ]
            for i, (x, y) in enumerate(features['landmarks_5']):
                cv2.circle(img, (int(x), int(y)), 4, colors[i], -1)
                
        # Draw 106-point landmarks
        if draw_106pt and features.get('landmarks_106'):
            for x, y in features['landmarks_106']:
                cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), -1)
                
        return img
    
    @staticmethod
    def draw_info(image: np.ndarray, 
                  features: Dict,
                  position: tuple = (10, 25)) -> np.ndarray:
        """
        Draw feature information text on an image.
        
        Args:
            image: Image to draw on (BGR format)
            features: Features dict from extract_features()
            position: Starting position for text (x, y)
            
        Returns:
            Image with info text drawn
        """
        img = image.copy()
        x, y = position
        line_height = 25
        
        # Detection score
        if features.get('det_score') is not None:
            text = f"Score: {features['det_score']:.2f}"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += line_height
        
        # Age
        if features.get('age') is not None:
            text = f"Age: {features['age']}"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += line_height
            
        # Gender
        if features.get('gender') is not None:
            text = f"Gender: {features['gender']}"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += line_height
            
        # Pose
        if features.get('pose') is not None:
            pose = features['pose']
            text = f"Pose: P:{pose['pitch']:.1f} Y:{pose['yaw']:.1f} R:{pose['roll']:.1f}"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y += line_height
            
        return img
    
    @staticmethod
    def draw_bbox(image: np.ndarray,
                  features: Dict,
                  color: tuple = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw face bounding box on an image.
        
        Args:
            image: Image to draw on (BGR format)
            features: Features dict from extract_features()
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with bounding box drawn
        """
        img = image.copy()
        
        if features.get('bbox'):
            x1, y1, x2, y2 = map(int, features['bbox'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
        return img
    
    @staticmethod
    def compare_embeddings(embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First face embedding (512-dim)
            embedding2: Second face embedding (512-dim)
            
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    @staticmethod
    def is_same_person(embedding1: np.ndarray,
                       embedding2: np.ndarray,
                       threshold: float = 0.4) -> bool:
        """
        Check if two embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (default 0.4)
            
        Returns:
            True if same person, False otherwise
        """
        similarity = FaceFeatures.compare_embeddings(embedding1, embedding2)
        return similarity >= threshold
    
    @staticmethod
    def get_summary(features: Dict) -> str:
        """
        Get a human-readable summary of face features.
        
        Args:
            features: Features dict from extract_features()
            
        Returns:
            Summary string
        """
        parts = []
        
        if features.get('det_score') is not None:
            parts.append(f"Score: {features['det_score']:.2f}")
            
        if features.get('age') is not None:
            parts.append(f"Age: {features['age']}")
            
        if features.get('gender') is not None:
            parts.append(f"Gender: {features['gender']}")
            
        if features.get('pose') is not None:
            pose = features['pose']
            parts.append(f"Pose: (P:{pose['pitch']:.0f}, Y:{pose['yaw']:.0f}, R:{pose['roll']:.0f})")
            
        return " | ".join(parts) if parts else "No features available"
