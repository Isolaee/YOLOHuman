"""
Person Re-identification using YOLO + Deep Embeddings
Detects humans and compares them across images using feature similarity
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
from pathlib import Path


class PersonReID:
    def __init__(self, yolo_model_path: str, similarity_threshold: float = 0.7):
        """
        Initialize Person Re-identification system
        
        Args:
            yolo_model_path: Path to YOLO model weights
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
        """
        self.similarity_threshold = similarity_threshold
        
        # Load YOLO for detection
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        
        # Load ResNet for feature extraction (remove classification head)
        print("Loading feature extraction model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard ReID size (height x width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Store known persons: {person_id: [embeddings]}
        self.known_persons = {}
        self.next_person_id = 1
        
        print(f"ReID system ready (device: {self.device})")
    
    def detect_persons(self, image_path: str) -> list:
        """
        Detect all persons in an image
        
        Returns:
            List of dicts with 'bbox', 'confidence', 'crop'
        """
        results = self.yolo(image_path, verbose=False)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                
                # Crop the person
                crop = image_rgb[y1:y2, x1:x2]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'crop': crop
                })
        
        return detections
    
    def extract_embedding(self, person_crop: np.ndarray) -> torch.Tensor:
        """Extract feature embedding from a person crop"""
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(person_crop)
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            embedding = self.feature_extractor(img_tensor)
            embedding = embedding.flatten()
            # L2 normalize
            embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding.cpu()
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def register_person(self, person_crop: np.ndarray, person_id: int = None) -> int:
        """
        Register a new person or add embedding to existing person
        
        Returns:
            person_id
        """
        embedding = self.extract_embedding(person_crop)
        
        if person_id is None:
            person_id = self.next_person_id
            self.next_person_id += 1
            self.known_persons[person_id] = []
        
        self.known_persons[person_id].append(embedding)
        return person_id
    
    def identify_person(self, person_crop: np.ndarray) -> tuple:
        """
        Try to identify a person against known persons
        
        Returns:
            (person_id, similarity) or (None, 0) if no match
        """
        if not self.known_persons:
            return None, 0.0
        
        query_embedding = self.extract_embedding(person_crop)
        
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, embeddings in self.known_persons.items():
            # Compare with all embeddings for this person (take max)
            for emb in embeddings:
                similarity = self.compute_similarity(query_embedding, emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
        
        if best_similarity >= self.similarity_threshold:
            return best_match_id, best_similarity
        
        return None, best_similarity
    
    def extract_persons(self, image_path: str, output_dir: str = None) -> list:
        """
        Step 1: Extract all persons from an image and optionally save crops
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save person crops (optional)
            
        Returns:
            List of person dicts with 'id', 'bbox', 'confidence', 'crop', 'crop_path', 'embedding'
        """
        image_name = Path(image_path).stem
        detections = self.detect_persons(image_path)
        
        persons = []
        for i, det in enumerate(detections):
            person = {
                'id': i + 1,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'crop': det['crop'],
                'crop_path': None,
                'embedding': self.extract_embedding(det['crop'])
            }
            
            # Save crop if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                crop_filename = f"{image_name}_person_{i+1}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cv2.cvtColor(det['crop'], cv2.COLOR_RGB2BGR))
                person['crop_path'] = crop_path
            
            persons.append(person)
        
        return persons
    
    def compare_persons(self, persons1: list, persons2: list) -> dict:
        """
        Step 2: Compare extracted persons from two sets
        
        Args:
            persons1: List of persons from first image (from extract_persons)
            persons2: List of persons from second image (from extract_persons)
            
        Returns:
            Dictionary with similarity matrix and matches
        """
        if not persons1 or not persons2:
            return {'matches': [], 'similarity_matrix': None, 'message': 'Not enough persons to compare'}
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(persons1), len(persons2)))
        matches = []
        
        for i, p1 in enumerate(persons1):
            for j, p2 in enumerate(persons2):
                similarity = self.compute_similarity(p1['embedding'], p2['embedding'])
                similarity_matrix[i, j] = similarity
                
                if similarity >= self.similarity_threshold:
                    matches.append({
                        'person1_id': p1['id'],
                        'person2_id': p2['id'],
                        'similarity': similarity,
                        'bbox1': p1['bbox'],
                        'bbox2': p2['bbox'],
                        'crop_path1': p1.get('crop_path'),
                        'crop_path2': p2.get('crop_path')
                    })
        
        # Sort matches by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'persons1_count': len(persons1),
            'persons2_count': len(persons2),
            'similarity_matrix': similarity_matrix,
            'matches': matches,
            'threshold': self.similarity_threshold
        }
    
    def compare_two_images(self, image1_path: str, image2_path: str, 
                          save_crops: bool = True, crops_dir: str = "test/crops") -> dict:
        """
        Full pipeline: Extract persons from both images, then compare them
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            save_crops: Whether to save extracted person crops
            crops_dir: Directory to save crops
            
        Returns:
            Dictionary with comparison results
        """
        # Step 1: Extract persons from image 1
        print(f"\n{'='*50}")
        print("STEP 1: EXTRACTING PERSONS")
        print('='*50)
        
        print(f"\nImage 1: {image1_path}")
        output_dir1 = os.path.join(crops_dir, "image1") if save_crops else None
        persons1 = self.extract_persons(image1_path, output_dir1)
        print(f"  Extracted {len(persons1)} person(s)")
        for p in persons1:
            print(f"    - Person {p['id']}: confidence={p['confidence']:.2f}", end="")
            if p['crop_path']:
                print(f", saved to: {p['crop_path']}")
            else:
                print()
        
        print(f"\nImage 2: {image2_path}")
        output_dir2 = os.path.join(crops_dir, "image2") if save_crops else None
        persons2 = self.extract_persons(image2_path, output_dir2)
        print(f"  Extracted {len(persons2)} person(s)")
        for p in persons2:
            print(f"    - Person {p['id']}: confidence={p['confidence']:.2f}", end="")
            if p['crop_path']:
                print(f", saved to: {p['crop_path']}")
            else:
                print()
        
        # Step 2: Compare extracted persons
        print(f"\n{'='*50}")
        print("STEP 2: COMPARING PERSONS")
        print('='*50)
        
        results = self.compare_persons(persons1, persons2)
        results['persons1'] = persons1
        results['persons2'] = persons2
        results['image1_path'] = image1_path
        results['image2_path'] = image2_path
        
        return results
    
    def visualize_comparison(self, results: dict, output_path: str = None):
        """Create visualization of comparison results using cropped person images"""
        persons1 = results.get('persons1', [])
        persons2 = results.get('persons2', [])
        
        if not persons1 or not persons2:
            print("No persons to visualize")
            return None
        
        # Prepare cropped images with labels
        crops1 = []
        crops2 = []
        target_height = 256  # Standard height for all crops
        
        for p in persons1:
            crop = cv2.cvtColor(p['crop'], cv2.COLOR_RGB2BGR)
            h, w = crop.shape[:2]
            scale = target_height / h
            crop = cv2.resize(crop, (int(w * scale), target_height))
            # Add label
            cv2.putText(crop, f"P{p['id']}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(crop, f"{p['confidence']:.2f}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            crops1.append(crop)
        
        for p in persons2:
            crop = cv2.cvtColor(p['crop'], cv2.COLOR_RGB2BGR)
            h, w = crop.shape[:2]
            scale = target_height / h
            crop = cv2.resize(crop, (int(w * scale), target_height))
            # Add label
            cv2.putText(crop, f"P{p['id']}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(crop, f"{p['confidence']:.2f}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            crops2.append(crop)
        
        # Create rows for each image's persons
        padding = 10
        row1_width = sum(c.shape[1] for c in crops1) + padding * (len(crops1) + 1)
        row2_width = sum(c.shape[1] for c in crops2) + padding * (len(crops2) + 1)
        max_width = max(row1_width, row2_width, 400)
        
        # Create image with space for labels and comparison info
        header_height = 40
        gap_height = 100  # Space between rows for drawing match lines
        total_height = header_height + target_height + gap_height + target_height + header_height + 50
        
        canvas = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
        
        # Draw header for Image 1
        cv2.putText(canvas, "Image 1 - Extracted Persons:", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Place crops from image 1
        x_offset = padding
        crop1_centers = []
        for crop in crops1:
            y_start = header_height
            y_end = y_start + target_height
            x_end = x_offset + crop.shape[1]
            canvas[y_start:y_end, x_offset:x_end] = crop
            crop1_centers.append((x_offset + crop.shape[1] // 2, y_end))
            x_offset = x_end + padding
        
        # Draw header for Image 2
        row2_y_start = header_height + target_height + gap_height
        cv2.putText(canvas, "Image 2 - Extracted Persons:", (10, row2_y_start - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Place crops from image 2
        x_offset = padding
        crop2_centers = []
        for crop in crops2:
            y_start = row2_y_start
            y_end = y_start + target_height
            x_end = x_offset + crop.shape[1]
            canvas[y_start:y_end, x_offset:x_end] = crop
            crop2_centers.append((x_offset + crop.shape[1] // 2, y_start))
            x_offset = x_end + padding
        
        # Draw match lines with similarity scores
        colors = [(0, 200, 0), (200, 0, 0), (0, 0, 200), (200, 200, 0), (200, 0, 200)]
        
        for idx, match in enumerate(results.get('matches', [])):
            color = colors[idx % len(colors)]
            p1_idx = match['person1_id'] - 1
            p2_idx = match['person2_id'] - 1
            
            if p1_idx < len(crop1_centers) and p2_idx < len(crop2_centers):
                pt1 = crop1_centers[p1_idx]
                pt2 = crop2_centers[p2_idx]
                
                # Draw line
                cv2.line(canvas, pt1, pt2, color, 3)
                
                # Add similarity label
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                label = f"Match: {match['similarity']:.2f}"
                cv2.putText(canvas, label, (mid_x - 50, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # If no matches, show all similarities in the gap area
        if not results.get('matches', []):
            sim_matrix = results.get('similarity_matrix')
            if sim_matrix is not None:
                y_pos = header_height + target_height + 30
                cv2.putText(canvas, "No matches above threshold. Similarities:", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                for i in range(len(persons1)):
                    for j in range(len(persons2)):
                        sim = sim_matrix[i, j]
                        y_pos += 20
                        cv2.putText(canvas, f"  P{i+1} <-> P{j+1}: {sim:.3f}", (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Add summary at bottom
        summary_y = total_height - 20
        threshold = results.get('threshold', 0.6)
        match_count = len(results.get('matches', []))
        summary = f"Threshold: {threshold} | Matches found: {match_count}"
        cv2.putText(canvas, summary, (10, summary_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"\nVisualization saved to: {output_path}")
        
        return canvas
    
    def compare_all_images(self, image_paths: list, save_crops: bool = True, 
                          crops_dir: str = "test/crops") -> dict:
        """
        Compare persons across ALL images and find matching identities
        
        Args:
            image_paths: List of image paths to process
            save_crops: Whether to save extracted person crops
            crops_dir: Directory to save crops
            
        Returns:
            Dictionary with all persons, matches, and identity groups
        """
        print(f"\n{'='*60}")
        print("STEP 1: EXTRACTING PERSONS FROM ALL IMAGES")
        print('='*60)
        
        # Extract persons from all images
        all_persons = []  # List of all persons with image info
        
        for img_idx, image_path in enumerate(image_paths):
            image_name = Path(image_path).stem
            print(f"\n[{img_idx + 1}/{len(image_paths)}] {Path(image_path).name}")
            
            output_dir = os.path.join(crops_dir, f"image_{img_idx + 1}") if save_crops else None
            persons = self.extract_persons(str(image_path), output_dir)
            
            print(f"  Extracted {len(persons)} person(s)")
            
            for p in persons:
                # Add image reference to each person
                p['image_idx'] = img_idx
                p['image_path'] = str(image_path)
                p['image_name'] = image_name
                p['global_id'] = f"img{img_idx + 1}_p{p['id']}"
                all_persons.append(p)
                
                print(f"    - {p['global_id']}: confidence={p['confidence']:.2f}", end="")
                if p['crop_path']:
                    print(f", saved to: {p['crop_path']}")
                else:
                    print()
        
        print(f"\nTotal persons extracted: {len(all_persons)}")
        
        # Step 2: Compare all persons against each other
        print(f"\n{'='*60}")
        print("STEP 2: COMPARING ALL PERSONS")
        print('='*60)
        
        # Build similarity matrix for all persons
        n = len(all_persons)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_similarity(all_persons[i]['embedding'], 
                                             all_persons[j]['embedding'])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Find all matches above threshold
        matches = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    matches.append({
                        'person1': all_persons[i],
                        'person2': all_persons[j],
                        'similarity': similarity_matrix[i, j]
                    })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Step 3: Group persons into identities using Union-Find
        print(f"\n{'='*60}")
        print("STEP 3: GROUPING MATCHING PERSONS INTO IDENTITIES")
        print('='*60)
        
        # Union-Find for grouping
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union all matching persons
        for match in matches:
            idx1 = all_persons.index(match['person1'])
            idx2 = all_persons.index(match['person2'])
            union(idx1, idx2)
        
        # Build identity groups
        groups = {}
        for i, person in enumerate(all_persons):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(person)
        
        # Convert to list and assign identity IDs
        identity_groups = []
        for identity_id, (root, members) in enumerate(groups.items(), 1):
            group = {
                'identity_id': identity_id,
                'members': members,
                'images': list(set(p['image_name'] for p in members)),
                'count': len(members)
            }
            identity_groups.append(group)
        
        # Sort by number of appearances (most common identities first)
        identity_groups.sort(key=lambda x: x['count'], reverse=True)
        
        # Print results
        print(f"\nFound {len(identity_groups)} unique identities:")
        for group in identity_groups:
            print(f"\n  Identity {group['identity_id']} (appears in {group['count']} detection(s)):")
            for member in group['members']:
                print(f"    - {member['global_id']} from {member['image_name']} "
                      f"(conf: {member['confidence']:.2f})")
        
        # Print matches
        print(f"\n{'='*60}")
        print(f"MATCHES (threshold >= {self.similarity_threshold})")
        print('='*60)
        
        if matches:
            for match in matches:
                p1, p2 = match['person1'], match['person2']
                print(f"  {p1['global_id']} <-> {p2['global_id']}: {match['similarity']:.3f}")
        else:
            print("  No matches found above threshold")
        
        return {
            'all_persons': all_persons,
            'similarity_matrix': similarity_matrix,
            'matches': matches,
            'identity_groups': identity_groups,
            'threshold': self.similarity_threshold,
            'image_paths': image_paths
        }
    
    def visualize_identities(self, results: dict, output_path: str = None):
        """
        Visualize all identity groups - showing which persons are the same
        """
        identity_groups = results.get('identity_groups', [])
        all_persons = results.get('all_persons', [])
        
        if not identity_groups:
            print("No identities to visualize")
            return None
        
        # Settings
        crop_height = 200
        padding = 10
        header_height = 30
        group_spacing = 40
        
        # Calculate dimensions
        max_group_width = 0
        total_height = padding
        
        for group in identity_groups:
            group_width = padding
            for member in group['members']:
                h, w = member['crop'].shape[:2]
                scale = crop_height / h
                group_width += int(w * scale) + padding
            max_group_width = max(max_group_width, group_width)
            total_height += header_height + crop_height + group_spacing
        
        max_width = max(max_group_width + 100, 600)
        
        # Create canvas
        canvas = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
        
        # Colors for different identities
        colors = [(0, 200, 0), (200, 0, 0), (0, 0, 200), (200, 200, 0), 
                  (200, 0, 200), (0, 200, 200), (100, 100, 200), (200, 100, 100)]
        
        y_offset = padding
        
        for group in identity_groups:
            color = colors[(group['identity_id'] - 1) % len(colors)]
            
            # Draw header
            header_text = f"Identity {group['identity_id']} - {group['count']} detection(s) across {len(group['images'])} image(s)"
            cv2.putText(canvas, header_text, (padding, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += header_height
            
            # Draw border around group
            group_width = padding
            for member in group['members']:
                h, w = member['crop'].shape[:2]
                scale = crop_height / h
                group_width += int(w * scale) + padding
            
            cv2.rectangle(canvas, (padding - 5, y_offset - 5), 
                         (group_width + 5, y_offset + crop_height + 5), color, 2)
            
            # Place member crops
            x_offset = padding
            for member in group['members']:
                crop = cv2.cvtColor(member['crop'], cv2.COLOR_RGB2BGR)
                h, w = crop.shape[:2]
                scale = crop_height / h
                crop_resized = cv2.resize(crop, (int(w * scale), crop_height))
                
                crop_w = crop_resized.shape[1]
                canvas[y_offset:y_offset + crop_height, x_offset:x_offset + crop_w] = crop_resized
                
                # Add label
                label = f"{member['global_id']}"
                cv2.putText(canvas, label, (x_offset + 5, y_offset + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, label, (x_offset + 5, y_offset + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                x_offset += crop_w + padding
            
            y_offset += crop_height + group_spacing
        
        # Add summary
        summary = f"Total: {len(all_persons)} persons detected, {len(identity_groups)} unique identities"
        cv2.putText(canvas, summary, (padding, total_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"\nIdentity visualization saved to: {output_path}")
        
        return canvas


def main():
    # Initialize ReID system
    reid = PersonReID(
        yolo_model_path="model/YOLOv8-HumanDetection/best.pt",
        similarity_threshold=0.6  # Adjust based on your needs
    )
    
    # Test directory
    test_dir = Path("test/img")
    # Exclude result images from comparison
    images = [f for f in test_dir.glob("*.jpg") if not f.stem.startswith("result") and not f.stem.startswith("reid")]
    images += [f for f in test_dir.glob("*.png") if not f.stem.startswith("result") and not f.stem.startswith("reid")]
    
    if len(images) < 1:
        print("No images found in test/img/")
        return
    
    print(f"\nFound {len(images)} image(s) in test/img/")
    for img in images:
        print(f"  - {img.name}")
    
    # Compare ALL images and find matching persons
    results = reid.compare_all_images(
        [str(img) for img in images],
        save_crops=True,
        crops_dir="test/crops"
    )
    
    # Visualize identity groups
    reid.visualize_identities(
        results,
        output_path="test/img/reid_identities.jpg"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Images processed: {len(images)}")
    print(f"Total persons detected: {len(results['all_persons'])}")
    print(f"Unique identities found: {len(results['identity_groups'])}")
    print(f"Matches above threshold ({results['threshold']}): {len(results['matches'])}")


if __name__ == "__main__":
    main()
