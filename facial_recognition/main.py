'''
Docstring for facial_recognition.main
Main module connecting person detection, facial recognition, and face feature analysis.
'''

import cv2
import os
from typing import List, Dict, Any, Optional

from .persons import detect_humans, detect_humans_batch, IMAGE_FOLDER_PATH
from .facial_recognition import FacialRecognition
from .face_features import FaceFeatures


class PersonFaceAnalyzer:
    """
    Complete pipeline for detecting persons and analyzing their faces.
    Connects: persons.py -> facial_recognition.py -> face_features.py
    """
    
    def __init__(self,
                 model_name: str = "buffalo_l",
                 providers: List[str] = None):
        """
        Initialize the analyzer with InsightFace model.
        
        Args:
            model_name: InsightFace model pack name
            providers: ONNX execution providers
        """
        print("Initializing PersonFaceAnalyzer...")
        self.face_rec = FacialRecognition(model_name=model_name, providers=providers)
        self.face_features = FaceFeatures()
        print("PersonFaceAnalyzer ready!")
    
    def analyze_image(self, 
                      image_path: str,
                      confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Analyze an image: detect persons, find faces, extract features.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Min confidence for person detection
            
        Returns:
            List of person results, each containing:
                - 'person_id': Index of the person
                - 'bbox': Person bounding box
                - 'confidence': Person detection confidence
                - 'crop': Cropped person image
                - 'faces': List of face results with features
        """
        # Step 1: Detect humans
        detections = detect_humans(image_path, confidence_threshold)
        
        results = []
        
        for i, det in enumerate(detections):
            person_result = {
                'person_id': i + 1,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'crop': det['crop'],
                'class_name': det['class_name'],
                'faces': []
            }
            
            # Step 2: Get faces from person crop
            faces = self.face_rec.get_faces_from_crop(det['crop'])
            
            # Step 3: Extract features from each face
            for j, face in enumerate(faces):
                features = self.face_features.extract_features(face)
                features['face_id'] = j + 1
                person_result['faces'].append(features)
            
            results.append(person_result)
        
        return results
    
    def analyze_folder(self,
                       image_folder: str = None,
                       confidence_threshold: float = 0.25) -> Dict[str, List[Dict]]:
        """
        Analyze all images in a folder.
        
        Args:
            image_folder: Path to folder (default: IMAGE_FOLDER_PATH)
            confidence_threshold: Min confidence for person detection
            
        Returns:
            Dictionary mapping image paths to their analysis results
        """
        if image_folder is None:
            image_folder = IMAGE_FOLDER_PATH
            
        # Find images
        image_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('result')
        ]
        
        all_results = {}
        
        for image_path in image_files:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            results = self.analyze_image(image_path, confidence_threshold)
            all_results[image_path] = results
            
            # Print summary
            total_faces = sum(len(p['faces']) for p in results)
            print(f"  Found {len(results)} person(s), {total_faces} face(s)")
        
        return all_results
    
    def draw_results(self,
                     image: Any,
                     results: List[Dict],
                     draw_person_bbox: bool = True,
                     draw_face_bbox: bool = True,
                     draw_landmarks: bool = True,
                     draw_info: bool = True) -> Any:
        """
        Draw detection and analysis results on an image.
        
        Args:
            image: Image (path or numpy array)
            results: Analysis results from analyze_image()
            draw_person_bbox: Draw person bounding boxes
            draw_face_bbox: Draw face bounding boxes
            draw_landmarks: Draw facial landmarks
            draw_info: Draw age/gender info
            
        Returns:
            Annotated image
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        for person in results:
            # Draw person bounding box
            if draw_person_bbox:
                x1, y1, x2, y2 = map(int, person['bbox'])
                has_face = len(person['faces']) > 0
                color = (0, 255, 0) if has_face else (0, 165, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"Person {person['person_id']}: {person['confidence']:.2f}"
                if has_face:
                    face = person['faces'][0]
                    if face.get('age') and face.get('gender'):
                        label += f" [{face['gender']}, {face['age']}]"
                        
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw face info on person crop region
            for face in person['faces']:
                px1, py1, px2, py2 = map(int, person['bbox'])
                
                if draw_face_bbox and face.get('bbox'):
                    # Face bbox is relative to person crop, adjust to full image
                    fx1, fy1, fx2, fy2 = map(int, face['bbox'])
                    cv2.rectangle(img, (px1 + fx1, py1 + fy1), 
                                 (px1 + fx2, py1 + fy2), (255, 0, 255), 2)
                
                if draw_landmarks and face.get('landmarks_5'):
                    for x, y in face['landmarks_5']:
                        cv2.circle(img, (px1 + int(x), py1 + int(y)), 3, (0, 255, 255), -1)
        
        return img
    
    def save_results(self,
                     image_path: str,
                     results: List[Dict],
                     output_dir: str = None) -> Dict[str, str]:
        """
        Save analysis results: annotated image, person crops, face crops.
        
        Args:
            image_path: Original image path
            results: Analysis results
            output_dir: Output directory (default: same as image)
            
        Returns:
            Dictionary of saved file paths
        """
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
            
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        saved_files = {}
        
        # Save annotated image
        img = cv2.imread(image_path)
        annotated = self.draw_results(img, results)
        annotated_path = os.path.join(output_dir, f"result_{base_name}.jpg")
        cv2.imwrite(annotated_path, annotated)
        saved_files['annotated'] = annotated_path
        
        # Save person crops
        saved_files['person_crops'] = []
        for person in results:
            crop_path = os.path.join(output_dir, f"{base_name}_person_{person['person_id']}.jpg")
            cv2.imwrite(crop_path, person['crop'])
            saved_files['person_crops'].append(crop_path)
        
        return saved_files
    
    def print_results(self, results: List[Dict]):
        """
        Print analysis results in a formatted way.
        
        Args:
            results: Analysis results from analyze_image()
        """
        for person in results:
            print(f"\nPerson {person['person_id']}:")
            print(f"  Confidence: {person['confidence']:.3f}")
            print(f"  Bounding box: {[int(x) for x in person['bbox']]}")
            print(f"  Faces found: {len(person['faces'])}")
            
            for face in person['faces']:
                print(f"\n  Face {face['face_id']}:")
                print(f"    {self.face_features.get_summary(face)}")
                
                if face.get('embedding') is not None:
                    print(f"    Embedding: {face['embedding'].shape} dimensional")


def main():
    """Main entry point for testing."""
    print("="*60)
    print("PERSON + FACE DETECTION + FEATURE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = PersonFaceAnalyzer()
    
    # Analyze all images in test folder
    all_results = analyzer.analyze_folder()
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    total_persons = 0
    total_faces = 0
    
    for image_path, results in all_results.items():
        print(f"\n{'='*60}")
        print(f"Image: {os.path.basename(image_path)}")
        print("="*60)
        
        analyzer.print_results(results)
        
        # Save annotated image
        img = cv2.imread(image_path)
        annotated = analyzer.draw_results(img, results)
        output_path = os.path.join(os.path.dirname(image_path), 
                                   f"result_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved: {output_path}")
        
        total_persons += len(results)
        total_faces += sum(len(p['faces']) for p in results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total persons detected: {total_persons}")
    print(f"Total faces analyzed: {total_faces}")
    print(f"Face detection rate: {100*total_faces/total_persons:.1f}%" if total_persons > 0 else "")


if __name__ == "__main__":
    main()
