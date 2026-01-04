'''
Docstring for facial_recognition.main
Main module connecting person detection, facial recognition, and face feature analysis.
'''

import cv2
import os
import numpy as np
from typing import List, Dict, Any, Optional

from .persons import detect_humans, detect_humans_batch, IMAGE_FOLDER_PATH
from .facial_recognition import FacialRecognition
from .face_features import FaceFeatures
from .face_matches import FaceMatcher, PersonCluster


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


def visualize_matches(clusters: List[PersonCluster], 
                      all_results: Dict[str, List[Dict]],
                      face_size: int = 120,
                      max_faces_per_row: int = 8) -> np.ndarray:
    """
    Create a visual representation of face matches.
    
    Args:
        clusters: List of PersonCluster objects
        all_results: Analysis results with crops
        face_size: Size to resize face thumbnails
        max_faces_per_row: Maximum faces per row
        
    Returns:
        Visualization image
    """
    # Build lookup for person crops
    crops_lookup = {}
    for image_path, persons in all_results.items():
        for person in persons:
            key = f"{image_path}:p{person['person_id']}"
            crops_lookup[key] = person['crop']
    
    # Calculate layout
    rows = []
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    for cluster_idx, cluster in enumerate(clusters):
        color = colors[cluster_idx % len(colors)]
        summary = cluster.get_summary()
        
        # Create header for this cluster
        header_height = 40
        num_faces = len(cluster.faces)
        row_width = min(num_faces, max_faces_per_row) * (face_size + 10) + 20
        row_width = max(row_width, 400)
        
        # Create rows of faces for this cluster
        face_rows = []
        current_row = []
        
        for face in cluster.faces:
            # Get the crop
            crop_key = f"{face.image_path}:p{face.person_id}"
            crop = crops_lookup.get(crop_key)
            
            if crop is None:
                continue
            
            # Extract face region from crop if available
            if face.features.get('bbox'):
                fx1, fy1, fx2, fy2 = map(int, face.features['bbox'])
                # Add padding
                h, w = crop.shape[:2]
                pad = 20
                fx1, fy1 = max(0, fx1 - pad), max(0, fy1 - pad)
                fx2, fy2 = min(w, fx2 + pad), min(h, fy2 + pad)
                face_img = crop[fy1:fy2, fx1:fx2]
            else:
                # Use top portion of person crop as face
                h, w = crop.shape[:2]
                face_img = crop[0:min(h, int(h*0.4)), :]
            
            # Resize to thumbnail
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (face_size, face_size))
                
                # Add border with cluster color
                face_img = cv2.copyMakeBorder(face_img, 3, 3, 3, 3, 
                                              cv2.BORDER_CONSTANT, value=color)
                
                # Add label
                img_name = os.path.basename(face.image_path)[:15]
                age = face.features.get('age', '?')
                cv2.putText(face_img, f"{img_name}", (5, face_size + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                current_row.append(face_img)
                
                if len(current_row) >= max_faces_per_row:
                    face_rows.append(current_row)
                    current_row = []
        
        if current_row:
            face_rows.append(current_row)
        
        # Create cluster visualization
        if face_rows:
            # Header
            header = np.zeros((header_height, row_width, 3), dtype=np.uint8)
            header[:] = (40, 40, 40)
            
            title = f"Person {summary['cluster_id']}: {summary['face_count']} face(s) in {summary['image_count']} image(s)"
            if summary['avg_age'] and summary['gender']:
                title += f" | {summary['gender']}, ~{summary['avg_age']}yo"
            
            cv2.putText(header, title, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            rows.append(header)
            
            # Face rows
            for face_row in face_rows:
                row_height = face_size + 20  # Extra space for border
                row_img = np.zeros((row_height, row_width, 3), dtype=np.uint8)
                row_img[:] = (30, 30, 30)
                
                x_offset = 10
                for face_img in face_row:
                    fh, fw = face_img.shape[:2]
                    y_start = 5
                    y_end = min(y_start + fh, row_height)
                    x_end = min(x_offset + fw, row_width)
                    actual_fh = y_end - y_start
                    actual_fw = x_end - x_offset
                    if actual_fh > 0 and actual_fw > 0:
                        row_img[y_start:y_end, x_offset:x_end] = face_img[:actual_fh, :actual_fw]
                    x_offset += fw + 10
                
                rows.append(row_img)
            
            # Add spacing between clusters
            spacer = np.zeros((10, row_width, 3), dtype=np.uint8)
            rows.append(spacer)
    
    if not rows:
        # No matches - create placeholder
        placeholder = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No faces detected", (100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return placeholder
    
    # Find max width and create final image
    max_width = max(r.shape[1] for r in rows)
    
    # Pad rows to same width
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_width:
            pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            pad[:] = (30, 30, 30)
            row = np.hstack([row, pad])
        padded_rows.append(row)
    
    return np.vstack(padded_rows)


def show_matches_window(clusters: List[PersonCluster],
                        all_results: Dict[str, List[Dict]],
                        window_name: str = "Face Matches"):
    """
    Show face matches in an OpenCV window.
    
    Args:
        clusters: List of PersonCluster objects
        all_results: Analysis results
        window_name: Window title
    """
    visualization = visualize_matches(clusters, all_results)
    
    # Resize if too large
    max_height = 900
    max_width = 1400
    h, w = visualization.shape[:2]
    
    if h > max_height or w > max_width:
        scale = min(max_height / h, max_width / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        visualization = cv2.resize(visualization, (new_w, new_h))
    
    cv2.imshow(window_name, visualization)
    print(f"\nVisualization window opened. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total persons detected: {total_persons}")
    print(f"Total faces analyzed: {total_faces}")
    print(f"Face detection rate: {100*total_faces/total_persons:.1f}%" if total_persons > 0 else "")
    
    # Face matching - find same persons across images
    print("\n" + "="*60)
    print("FACE MATCHING - FINDING SAME PERSONS")
    print("="*60)
    
    matcher = FaceMatcher(similarity_threshold=0.4)
    matcher.add_from_results(all_results)
    
    print(f"\nTotal faces for matching: {len(matcher.faces)}")
    
    # Cluster faces by identity
    clusters = matcher.cluster_faces()
    matcher.print_clusters()
    
    # Find persons appearing in multiple images
    print("="*60)
    print("CROSS-IMAGE MATCHES (Same person in multiple images)")
    print("="*60)
    
    cross_matches = matcher.find_same_person_across_images()
    
    if cross_matches:
        print(f"\nFound {len(cross_matches)} person(s) appearing in multiple images:\n")
        for cluster in cross_matches:
            summary = cluster.get_summary()
            images = [os.path.basename(img) for img in summary['images']]
            print(f"  Person {summary['cluster_id']}: appears in {summary['image_count']} images")
            print(f"    Age: ~{summary['avg_age']}, Gender: {summary['gender']}")
            print(f"    Images: {', '.join(images)}")
            print()
    else:
        print("\nNo persons found appearing in multiple images.")
    
    # Show visual representation
    print("\n" + "="*60)
    print("VISUAL REPRESENTATION")
    print("="*60)
    
    show_matches_window(clusters, all_results, "Face Matches - Same Persons")
    
    print("="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
