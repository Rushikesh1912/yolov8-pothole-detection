import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import torch

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Pothole Detection System",
    layout="wide"
)

MODEL_PATH = r"C:\Users\rushi\OneDrive\Desktop\yolov8-pothole-detection\datasets\runs\detect\train2\weights\best.pt"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
source_type = st.sidebar.radio(
    "Select Input Type",
    ("Image", "Video", "Webcam")
)

st.sidebar.markdown("---")
st.sidebar.write("CUDA Available:", torch.cuda.is_available())

# ---------------- TITLE ----------------
st.title("🕳️ Pothole Detection using YOLOv8")
st.markdown("Real-world pothole detection on images, videos, and live webcam feed.")

# ---------------- IMAGE DETECTION ----------------
if source_type == "Image":
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        with st.spinner("Detecting potholes..."):
            results = model.predict(
                img_array,
                conf=confidence,
                device=0
            )

        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detected Potholes", use_column_width=True)

# ---------------- VIDEO DETECTION ----------------
elif source_type == "Video":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    frame,
                    conf=confidence,
                    device=0
                )

                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame, channels="RGB")

        cap.release()
        os.remove(video_path)

# ---------------- WEBCAM DETECTION ----------------
elif source_type == "Webcam":
    run = st.checkbox("Start Webcam")
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to access webcam.")
                break

            results = model.predict(
                frame,
                conf=confidence,
                device=0
            )

            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB")

        cap.release()
