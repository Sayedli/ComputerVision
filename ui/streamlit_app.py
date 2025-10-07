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


st.set_page_config(page_title="Face Recognition", layout="wide")

# Cozy warm CSS theme
st.markdown(
    """
    <style>
      :root {
        --bg: #fff7ed;          /* warm peach */
        --panel: #fff3e0;       /* light peach */
        --accent: #f59e0b;      /* amber */
        --accent-2: #f97316;    /* orange */
        --text: #3f2d20;        /* cocoa */
        --muted: #8c6d5a;       /* latte */
        --border: #f2d6b3;      /* sand */
      }
      .stApp { background: radial-gradient(1200px 600px at 10% -10%, #fff1df 0%, var(--bg) 40%, var(--bg) 100%); }
      .cozy-header {
        padding: 1.25rem 1rem 0.25rem;
        color: var(--text);
      }
      .cozy-subtitle { color: var(--muted); margin-top: -0.5rem; }
      .cozy-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.08);
      }
      .cozy-badge { color: white; background: var(--accent); padding: 0.2rem 0.5rem; border-radius: 999px; font-size: 0.8rem; }
      .cozy-hr { border: none; height: 1px; background: var(--border); margin: 1rem 0; }
      .cozy-footer { color: var(--muted); font-size: 0.9rem; padding: 0.5rem 0 1rem; }
      .cozy-table td, .cozy-table th { border-bottom: 1px solid var(--border) !important; }
      button[kind="secondary"] { border-color: var(--accent) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cozy-header">
      <h1>😊 Face Recognition</h1>
      <p class="cozy-subtitle">A warm little tool to label faces you know. Upload a photo or take a snapshot — we’ll handle the rest.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ✨ Settings")
    st.caption("Pick your model file and detection preferences.")
    knn_path_str = st.text_input("Model path", value=str(KNN_PATH), help="Path to trained knn.joblib")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        detector_model = st.selectbox("Detector", options=["hog", "cnn"], index=0)
    with col_m2:
        upsample = st.number_input("Upsample", min_value=0, max_value=2, value=1, step=1)
    threshold = st.slider("Unknown threshold", min_value=0.3, max_value=1.0, value=0.6, step=0.01, help="Lower = stricter")
    run_button = st.button("🔄 Load / Reload Model", use_container_width=True)
    st.markdown("<hr class='cozy-hr' />", unsafe_allow_html=True)
    with st.expander("About this app"):
        st.write("This cozy UI wraps a KNN classifier trained on 128‑D face embeddings from the `face_recognition` library.")

if 'knn' not in st.session_state or run_button:
    try:
        st.session_state.knn = load_knn(Path(knn_path_str))
        st.success("Loaded KNN model")
    except Exception as e:
        st.session_state.knn = None
        st.error(f"Failed to load KNN: {e}")

tab1, tab2 = st.tabs(["📷 Upload Image", "🎞️ Webcam Snapshot"])


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
    st.markdown("<div class='cozy-panel'>", unsafe_allow_html=True)
    st.subheader("Recognize from an Image")
    file = st.file_uploader("Drop a photo here or browse", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is not None:
        img = Image.open(io.BytesIO(file.read()))
        out = process_image(img)
        st.image(out["image"], caption="Annotated", use_column_width=True)
        if out["results"]:
            st.success(f"Detected {len(out['results'])} face(s)")
            st.dataframe(
                [{"label": r["label"], "distance": r["distance"], "confidence": r["confidence"]} for r in out["results"]],
                use_container_width=True,
            )
        else:
            st.info("No faces found — try a clearer, front‑facing photo.")
    else:
        st.caption("Tip: wider, well‑lit faces work best.")
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='cozy-panel'>", unsafe_allow_html=True)
    st.subheader("Take a Snapshot")
    st.caption("The photo stays in your browser session.")
    cam_img = st.camera_input("Click to capture")
    if cam_img is not None:
        img = Image.open(cam_img)
        out = process_image(img)
        st.image(out["image"], caption="Annotated", use_column_width=True)
        if out["results"]:
            st.success(f"Detected {len(out['results'])} face(s)")
            st.dataframe(
                [{"label": r["label"], "distance": r["distance"], "confidence": r["confidence"]} for r in out["results"]],
                use_container_width=True,
            )
        else:
            st.info("No faces found — try adjusting lighting or distance.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='cozy-footer'>Made with ☕ and 🍪 — enjoy!</div>", unsafe_allow_html=True)
