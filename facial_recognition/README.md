# Facial Recognition Module

Facial recognition using MediaPipe Face Mesh (468 landmarks).

## Features

- **Face Detection**: Detect faces in images
- **468 Facial Landmarks**: Extract detailed face mesh
- **Face Embeddings**: Generate embeddings for face comparison
- **Multi-Image Comparison**: Find same faces across multiple images
- **Identity Grouping**: Cluster matching faces into identities

## Installation

```bash
pip install mediapipe opencv-python numpy
```

## Usage

### Basic Usage

```python
from facial_recognition import FacialRecognition

# Initialize
face_rec = FacialRecognition(similarity_threshold=0.6)

# Extract faces from an image
faces = face_rec.extract_faces("image.jpg", output_dir="crops")

# Compare faces across multiple images
results = face_rec.compare_all_images(
    ["image1.jpg", "image2.jpg", "image3.jpg"],
    save_crops=True
)

# Visualize identity groups
face_rec.visualize_identities(results, output_path="identities.jpg")
```

### Register and Identify Faces

```python
# Register a known face
face_rec.register_face(face_crop, name="John")

# Later, identify an unknown face
face_id, name, similarity = face_rec.identify_face(unknown_crop)
```

### Visualize Face Landmarks

```python
# Draw 468-point face mesh on image
face_rec.visualize_face_landmarks("face.jpg", output_path="face_mesh.jpg")
```

## MediaPipe Landmarks

The system uses 468 facial landmarks including:

| Region | Landmark Count |
|--------|----------------|
| Face contour | 36 |
| Left eye | 8 |
| Right eye | 8 |
| Left eyebrow | 5 |
| Right eyebrow | 5 |
| Nose | 8 |
| Mouth | 11 |
| Iris | 10 (with refinement) |

## Output

- **Face crops**: Saved to `facial_crops/image_N/`
- **Identity visualization**: Shows grouped faces by identity
- **Face mesh visualization**: Shows 468 landmarks on face
