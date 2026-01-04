'''
Docstring for facial_recognition.persons
Module for finding persons in images and videos.
'''

from ultralytics import YOLO
import cv2
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to YOLOv8 model weights for human detection
MODEL_PATH = os.path.join(SCRIPT_DIR, "../model/YOLOv8-HumanDetection/best.pt")

#path to test images
IMAGE_FOLDER_PATH = os.path.join(SCRIPT_DIR, "../test/img")

# Load the YOLOv8 model
print("Loading YOLOv8 Human Detection Model...")
model = YOLO(MODEL_PATH)


def detect_humans(image_path, confidence_threshold=0.25):
    """
    Detect humans in an image and return detection data.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score (default 0.25)
    
    Returns:
        List of dictionaries, each containing:
            - 'bbox': [x1, y1, x2, y2] bounding box coordinates
            - 'confidence': detection confidence score
            - 'crop': cropped image of the detected human (numpy array)
            - 'class_name': detected class name
    """
    # Read the original image for cropping
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run inference
    results = model(image_path)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            confidence = box.conf[0].item()
            
            # Skip low confidence detections
            if confidence < confidence_threshold:
                continue
            
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            # Extract crop of detected human
            x1, y1, x2, y2 = map(int, coords)
            # Clamp coordinates to image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2].copy()
            
            detections.append({
                'bbox': coords,
                'confidence': confidence,
                'crop': crop,
                'class_name': class_name
            })
    
    return detections


def detect_humans_batch(image_folder=None):
    """
    Detect humans in all images in a folder.
    
    Args:
        image_folder: Path to folder containing images (default: IMAGE_FOLDER_PATH)
    
    Returns:
        Dictionary mapping image paths to their detection lists
    """
    if image_folder is None:
        image_folder = IMAGE_FOLDER_PATH
    
    # Find images with full paths
    image_files = [
        os.path.join(image_folder, f) 
        for f in os.listdir(image_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('result_')
    ]
    
    all_detections = {}
    
    for image_path in image_files:
        print(f"Processing: {image_path}")
        detections = detect_humans(image_path)
        all_detections[image_path] = detections
        print(f"  Found {len(detections)} human(s)")
    
    return all_detections


# Example usage when run directly
if __name__ == "__main__":
    results = detect_humans_batch()
    
    for image_path, detections in results.items():
        print(f"\n{os.path.basename(image_path)}: {len(detections)} detection(s)")
        for i, det in enumerate(detections):
            print(f"  [{i+1}] confidence: {det['confidence']:.2f}, bbox: {det['bbox']}")
            # det['crop'] contains the cropped image as numpy array
            # You can save it, process it further, etc.
            # Example: cv2.imwrite(f"crop_{i}.jpg", det['crop'])