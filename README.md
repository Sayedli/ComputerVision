# ComputerVision — Face Recognition MVP

Lightweight face recognition prototype using Python, OpenCV, and the `face_recognition` library (dlib). It supports quick data collection, encoding faces to embeddings, training a simple KNN classifier, and recognizing faces from the webcam or still images.

## Features
- Register faces via webcam and save cropped samples per person
- Encode dataset images into 128‑D face embeddings
- Train a small KNN classifier on embeddings
- Live webcam recognition with labels and rough confidence
- One‑off recognition for a single image

## Requirements
- Python 3.8+
- macOS/Linux/Windows
- Packages listed in `requirements.txt`:
  - `face-recognition`, `opencv-python`, `numpy`, `scikit-learn`, `joblib`, `imutils`, `tqdm`

Note: `face-recognition` depends on `dlib`. If prebuilt wheels are unavailable for your platform/Python version, you may need build tools installed (e.g., CMake, a C++ compiler). On macOS, ensure Command Line Tools are installed. On Linux, install build essentials and CMake.

## Install
1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

If `face-recognition`/`dlib` fails to install, consult platform-specific instructions for building dlib from source (ensure CMake and a C++ toolchain are available).

## Project Layout
```
fr.py                # Thin CLI entry
fr/                  # Package with implementation
  cli.py             # CLI wiring
  paths.py           # Project paths and dirs
  utils.py           # Helpers (drawing, confidence, I/O utils)
  encode.py          # Dataset → embeddings logic
  train.py           # KNN training utilities
  recognize.py       # Single-image recognition
  webcam.py          # Live webcam recognition
  register.py        # Webcam data collection
dataset/             # Person folders with face images (created automatically)
  Alice/ *.jpg
  Bob/   *.jpg
models/              # Saved artifacts (created automatically)
  encodings.npz      # Embeddings (X) and labels (y)
  knn.joblib         # Trained KNN classifier
```

## Usage
The CLI is exposed by `fr.py`.

1) Register a person via webcam (optional but handy for collecting samples):
```
python fr.py register --name "Alice" --num 20
```
This captures ~20 face crops to `dataset/Alice/`.

2) Encode dataset images into embeddings:
```
python fr.py encode
```
Writes `models/encodings.npz` containing `X` (N×128 embeddings) and `y` (labels).

3) Train a KNN classifier on the embeddings:
```
python fr.py train
```
Writes `models/knn.joblib`.

4a) Live recognition from webcam:
```
python fr.py webcam
```

4b) Recognize a single image:
```
python fr.py recognize --image path/to/photo.jpg
```

## Code Structure
- `fr.cli`: argparse subcommands; forwards to modules.
- `fr.encode`: walks `dataset/`, detects faces, computes 128‑D encodings, saves NPZ.
- `fr.train`: loads encodings and trains `KNeighborsClassifier` (also exposes a pure `train_knn_from_arrays` for smoke tests).
- `fr.recognize`: classifies faces in a single image using the trained model.
- `fr.webcam`: live loop for webcam recognition (scaling + frame skipping for speed).
- `fr.register`: collects face crops via webcam and saves to `dataset/<name>/`.

## Developer Smoke Tests
This repo does not include a formal test suite. For a quick no‑I/O sanity check of training logic:

```
python3 scripts/smoke_train.py
```

It synthesizes two clusters in 128‑D, trains KNN via `fr.train.train_knn_from_arrays`, and reports accuracy.

## Running Pytest
Basic unit tests cover pure utilities and training helpers (no CV dependencies).

1) Install pytest (ideally in your venv):
```
pip install pytest
```

2) Run tests:
```
pytest
```

Notes:
- Tests intentionally avoid importing heavy modules (`face_recognition`, `cv2`).
- If you want broader tests (e.g., encode/recognize), ensure those deps are installed and we can add integration tests.

## Tips
- Data quality: Use clear, front-facing images with varied lighting/angles per person.
- Class balance: Aim for similar numbers of samples per person to help KNN.
- Webcam index: If the default webcam fails, try `OPENCV_VIDEOIO_PRIORITY_MSMF=0` on Windows or specify a different index in code if you add a flag.
- Performance: Encoding runs on CPU. More images = more time. Use smaller input images when possible.

## Troubleshooting
- ImportError: `No module named dlib` or `face_recognition`: Ensure dependencies are installed; install CMake and a C++ compiler toolchain if building from source.
- OpenCV camera issues: Close other apps using the camera; try another backend or device index.
- Empty encodings: Ensure faces are detected in your dataset images; remove images without faces or extreme occlusions.

## License
Proprietary/Private unless otherwise specified by the repository owner.
