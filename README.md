# YOLOHuman

Human detection and facial recognition project using YOLOv8 and InsightFace.

## Description

This project provides a complete pipeline for:
- **Human Detection** - Detect persons in images using YOLOv8
- **Facial Recognition** - Extract faces from detected persons using InsightFace
- **Face Features** - Analyze age, gender, pose, and landmarks
- **Face Matching** - Find the same person across multiple images using embedding similarity

## Setup

1. Create and activate virtual environment:
```powershell
python -m venv YOLOHuman
.\YOLOHuman\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install ultralytics insightface onnxruntime opencv-python numpy
```

## Usage

### Full Pipeline (Person Detection + Face Analysis + Matching)

```powershell
python -m facial_recognition.main
```

This will:
- Detect all persons in images from `test/img/`
- Extract and analyze faces (age, gender, pose)
- Match faces to find the same person across images
- Display a visual representation of matches

### Basic Human Detection Only

```powershell
python test_yolo.py
```

### Programmatic Usage

```python
from facial_recognition.main import PersonFaceAnalyzer
from facial_recognition.face_matches import FaceMatcher

# Initialize analyzer
analyzer = PersonFaceAnalyzer()

# Analyze single image
results = analyzer.analyze_image("path/to/image.jpg")
for person in results:
    print(f"Person {person['person_id']}: {person['confidence']:.2f}")
    for face in person['faces']:
        print(f"  Age: {face['age']}, Gender: {face['gender']}")

# Analyze folder and find matching faces
all_results = analyzer.analyze_folder("path/to/images/")

matcher = FaceMatcher(similarity_threshold=0.4)
matcher.add_from_results(all_results)
clusters = matcher.cluster_faces()

# Find same person across images
for cluster in matcher.find_same_person_across_images():
    print(f"Person appears in {cluster.image_count} images")
```

## Project Structure

```
YOLOHuman/
├── facial_recognition/
│   ├── __init__.py
│   ├── main.py              # Main pipeline orchestrator
│   ├── persons.py           # YOLO human detection
│   ├── facial_recognition.py # InsightFace face extraction
│   ├── face_features.py     # Feature extraction (age, gender, etc.)
│   └── face_matches.py      # Face similarity matching
├── model/
│   └── YOLOv8-HumanDetection/
│       ├── best.pt          # Best model weights
│       └── last.pt          # Last epoch weights
├── test/
│   ├── img/                 # Test images
│   ├── crops/               # Detected person crops
│   └── facial_crops/        # Extracted face crops
├── test_yolo.py             # Basic YOLO test script
└── README.md
```

## Features

| Feature | Description |
|---------|-------------|
| Person Detection | YOLOv8 model trained for human detection |
| Face Detection | InsightFace buffalo_l model |
| Age Estimation | Estimated age from facial features |
| Gender Classification | Male/Female classification |
| Face Embeddings | 512-dimensional vectors for face matching |
| Pose Estimation | Head pitch, yaw, roll angles |
| Landmarks | 5-point and 106-point facial landmarks |
| Cross-Image Matching | Find same person across multiple images |

## Output

- **Annotated images**: `test/img/result_*.jpg` - Bounding boxes with age/gender
- **Person crops**: `test/crops/` - Cropped images of detected persons
- **Face crops**: `test/facial_crops/` - Extracted face regions
- **Visual match window**: Interactive display of matched faces

## Credits

- YOLOv8 Human Detection model: Based on [YOLOv8-HumanDetection](https://github.com/YOLOv8-HumanDetection) repository
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis library

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

The pre-trained model in `model/YOLOv8-HumanDetection/` is also licensed under GPL-3.0.
