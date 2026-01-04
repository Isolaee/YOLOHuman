"""
Test YOLOv8 Human Detection Model on test images
"""
from ultralytics import YOLO
import cv2
import os

# Path to model weights
model_path = "model/YOLOv8-HumanDetection/best.pt"

# Path to test image
image_path = "test/img/IMG_20260104_131721888.jpg"

# Load the YOLO model
print("Loading YOLO model...")
model = YOLO(model_path)

# Run inference on the image
print(f"Running inference on: {image_path}")
results = model(image_path)

# Process results
for result in results:
    # Get detection info
    boxes = result.boxes
    
    print(f"\nDetections found: {len(boxes)}")
    
    # Print details for each detection
    for i, box in enumerate(boxes):
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        coords = box.xyxy[0].tolist()
        
        print(f"  Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
        print(f"    Bounding box: {coords}")
    
    # Save result image with bounding boxes
    output_path = "test/img/result.jpg"
    result.save(filename=output_path)
    print(f"\nResult saved to: {output_path}")
    
    # Display the result (opens a window)
    result.show()

print("\nDone!")
