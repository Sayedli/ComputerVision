# ComputerVision — Face Recognition MVP

A small, practical face-recognition toolset with a CLI and two simple UIs (Streamlit and Django). Uses `face_recognition` (dlib), OpenCV, and scikit‑learn (KNN).

## TL;DR (3 commands)

Docker (recommended)
```
docker compose build
docker compose run --rm fr fr encode && docker compose run --rm fr fr train
docker compose up ui   # open http://localhost:8501
```

Conda (local install)
```
conda create -n cv311 python=3.11 -y && conda activate cv311 && conda install -y -c conda-forge dlib face_recognition opencv numpy scikit-learn joblib imutils tqdm streamlit pillow
python3 fr.py encode && python3 fr.py train
streamlit run ui/streamlit_app.py   # then load models/knn.joblib in sidebar
```

## Quick Start

Option A — Docker (recommended)
- Build once:
  - `docker compose build`
- Optional: copy sample images into the dataset structure
  - Put images under `samples/<Person>/...`
  - `docker compose run --rm fr python3 scripts/prepare_samples.py`
- Encode + train:
  - `docker compose run --rm fr fr encode`
  - `docker compose run --rm fr fr train`
- Run a UI:
  - Streamlit: `docker compose up ui` → open http://localhost:8501
  - Django:    `docker compose up web` → open http://localhost:8000

Option B — Local install (Conda easiest)
- Create env and install prebuilt packages:
  - `conda create -n cv311 python=3.11 -y && conda activate cv311`
  - `conda install -y -c conda-forge dlib face_recognition opencv numpy scikit-learn joblib imutils tqdm streamlit pillow`
- Prepare data (optional): `python3 scripts/prepare_samples.py`
- Encode + train: `python3 fr.py encode && python3 fr.py train`
- Run a UI:
  - Streamlit: `streamlit run ui/streamlit_app.py` (load `models/knn.joblib` in the sidebar)
  - Django:    `python3 manage.py runserver` → http://127.0.0.1:8000

Tip (pip users on macOS/ARM): Building dlib locally can be painful. Prefer Docker or Conda.

## CLI Cheatsheet
- Register via webcam: `python3 fr.py register --name "Alice" --num 20`
- Encode dataset → embeddings: `python3 fr.py encode`
- Train KNN: `python3 fr.py train`
- Recognize image: `python3 fr.py recognize --image path/to/photo.jpg`
- Live webcam: `python3 fr.py webcam`

You can also install the console entry and use `fr` directly:
- `pip install -e .`
- `fr encode`, `fr train`, `fr recognize --image ...`

## Project Structure
- `fr.py` — thin CLI entry (calls the `fr/` package)
- `fr/`
  - `cli.py` — argparse subcommands
  - `paths.py` — directories and model paths
  - `utils.py` — drawing, confidence, helpers
  - `encode.py` — dataset → embeddings
  - `train.py` — KNN training utilities
  - `recognize.py` — single-image recognition
  - `webcam.py` — live webcam loop
  - `register.py` — collect face crops via webcam
- `ui/streamlit_app.py` — simple browser UI
- `manage.py`, `web/`, `recognition/` — minimal Django UI
- `dataset/` — your images, organized as `dataset/<Person>/*.jpg`
- `models/` — saved artifacts: `encodings.npz`, `knn.joblib`
- `scripts/prepare_samples.py` — copy `samples/<Person>` into `dataset/`

## Troubleshooting
- dlib build fails on macOS/ARM: use Docker or Conda.
- No faces detected: better lighting, frontal images; try higher `--upsample`.
- Webcam blocked: grant camera permission or use the Upload tab/UI.
- Model not loading in UIs: ensure `models/knn.joblib` exists; retrain after adding new data.

## Tests (optional)
- Install pytest: `pip install pytest`
- Run: `pytest` (covers pure helpers + training logic)
- Quick smoke: `python3 scripts/smoke_train.py`

## License
Private/Proprietary (unless the repository owner specifies otherwise).
