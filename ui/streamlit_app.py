#!/usr/bin/env python3
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Dict

import numpy as np
import streamlit as st
from PIL import Image
import cv2
import face_recognition

from fr.paths import KNN_PATH
from fr.recognize import load_knn
from fr.utils import draw_box_with_label, distance_to_confidence


st.set_page_config(page_title="Face Recognition MVP", layout="wide")
st.title("Face Recognition MVP – Streamlit UI")

with st.sidebar:
    st.header("Settings")
    knn_path_str = st.text_input("KNN model path", value=str(KNN_PATH))
    detector_model = st.selectbox("Detector model", options=["hog", "cnn"], index=0)
    upsample = st.number_input("Upsample", min_value=0, max_value=2, value=1, step=1)
    threshold = st.slider("Unknown threshold (distance)", min_value=0.3, max_value=1.0, value=0.6, step=0.01)
    run_button = st.button("Load Model")

if 'knn' not in st.session_state or run_button:
    try:
        st.session_state.knn = load_knn(Path(knn_path_str))
        st.success("Loaded KNN model")
    except Exception as e:
        st.session_state.knn = None
        st.error(f"Failed to load KNN: {e}")

tab1, tab2 = st.tabs(["Upload Image", "Webcam Snapshot"])


def process_image(pil_img: Image.Image) -> Dict:
    if st.session_state.knn is None:
        raise RuntimeError("Model not loaded. Use 'Load Model' in the sidebar.")

    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=int(upsample), model=detector_model)
    results: List[Dict] = []
    if boxes:
        encs = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
        preds = st.session_state.knn.predict(encs)
        dists, _ = st.session_state.knn.kneighbors(encs, n_neighbors=1, return_distance=True)
        dists = dists.reshape(-1)
        for box, pred_label, dist in zip(boxes, preds, dists):
            conf = distance_to_confidence(float(dist), float(threshold))
            label = pred_label if dist <= threshold else "Unknown"
            results.append({
                "box": box,
                "label": str(label),
                "distance": float(dist),
                "confidence": float(conf),
            })
            display_label = f"{label} {conf:.2f}"
            draw_box_with_label(bgr, box, display_label)

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return {"image": out_rgb, "results": results}


with tab1:
    st.subheader("Recognize from Uploaded Image")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is not None:
        img = Image.open(io.BytesIO(file.read()))
        out = process_image(img)
        st.image(out["image"], caption="Annotated", use_column_width=True)
        if out["results"]:
            st.success(f"Detected {len(out['results'])} face(s)")
            st.dataframe(
                [{"label": r["label"], "distance": r["distance"], "confidence": r["confidence"]} for r in out["results"]]
            )
        else:
            st.info("No faces found.")


with tab2:
    st.subheader("Recognize from Webcam Snapshot")
    cam_img = st.camera_input("Take a picture")
    if cam_img is not None:
        img = Image.open(cam_img)
        out = process_image(img)
        st.image(out["image"], caption="Annotated", use_column_width=True)
        if out["results"]:
            st.success(f"Detected {len(out['results'])} face(s)")
            st.dataframe(
                [{"label": r["label"], "distance": r["distance"], "confidence": r["confidence"]} for r in out["results"]]
            )
        else:
            st.info("No faces found.")

