# YOLOHuman

Human detection project using YOLOv8.

## Description

This project provides a simple interface to test YOLOv8 human detection on images. The model is based on the [YOLOv8-HumanDetection](https://github.com/YOLOv8-HumanDetection) repository.

## Setup

1. Create and activate virtual environment:
```powershell
python -m venv YOLOHuman
.\YOLOHuman\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install ultralytics
```

## Usage

Run detection on test images:
```powershell
python test_yolo.py
```

The script will:
- Load the pre-trained YOLOv8 model from `model/YOLOv8-HumanDetection/best.pt`
- Run inference on images in `test/img/`
- Save annotated results to `test/img/result.jpg`

## Project Structure

```
YOLOHuman/
├── model/
│   └── YOLOv8-HumanDetection/    # Pre-trained model (GPL-3.0)
│       ├── best.pt               # Best model weights
│       ├── last.pt               # Last epoch weights
│       └── ...
├── test/
│   └── img/                      # Test images
├── test_yolo.py                  # Test script
└── README.md
```

## Credits

- YOLOv8 Human Detection model: Based on [YOLOv8-HumanDetection](https://github.com/YOLOv8-HumanDetection) repository
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

The pre-trained model in `model/YOLOv8-HumanDetection/` is also licensed under GPL-3.0.
