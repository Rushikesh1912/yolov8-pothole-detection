🚧 Pothole Detection System using YOLOv8 & Streamlit

A complete end-to-end computer vision project for detecting road potholes using YOLOv8, trained on a custom dataset and deployed with a Streamlit web application supporting image, video, and real-time inference.

📌 Project Overview

Road potholes are a major cause of vehicle damage and traffic accidents. This project aims to automatically detect potholes from images and videos using deep learning, enabling applications in:
Smart city infrastructure monitoring
Road safety systems
Autonomous driving assistance
Municipal maintenance automation


🎥 Real-World Pothole Detection Demo (Model Output)

▶️ **Click below to watch the pothole detection demo video**

[![Pothole Detection Demo](assets/img-344_jpg.rf.dd4eebb8836b8.jpg)](assets/pothole_video%20(1).mp4)

Description: 
This video demonstrates real-time pothole detection using the trained YOLOv8 model.  
Bounding boxes and confidence scores are overlaid on real road footage.


✨ Key Features
✅ Custom YOLOv8 object detection model
✅ Trained on real-world pothole images with bounding box annotations
✅ Image, video, and real-time inference support
✅ GPU-accelerated training (NVIDIA CUDA)
✅ Evaluation using Precision, Recall, mAP@50, mAP@50-95
✅ Interactive Streamlit web application
✅ Deployment-ready architecture
✅ Model weights kept private (industry best practice)

Image upload & detection
Video upload & frame-by-frame detection
Bounding boxes with confidence scores

✅ GPU-accelerated inference (CUDA supported)
🧠 Model Details
| Item | Description |
|-----|------------|
| Architecture | YOLOv8m |
| Task | Object Detection |
| Classes | 1 (pothole) |
| Image Size | 640 × 640 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Scheduler | Cosine LR |
| Early Stopping | Enabled (patience = 10) |


📊 Model Performance (Validation Set)
| Metric | Score |
|------|------|
| Precision | 0.83 |
| Recall | 0.73 |
| mAP@50 | 0.81 |
| mAP@50–95 | 0.53 |


These results indicate strong detection accuracy with reliable generalization to unseen road conditions.

📁 Project Structure
yolov8-pothole-detection/
│
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── pothole/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── pothole/data.yaml              # Dataset configuration
│
└── runs/
    └── detect/
        └── train2/
            └── weights/
                └── best.pt        # Trained YOLOv8 model

🚀 Streamlit Web Application
The Streamlit app allows users to:
Upload an image and detect potholes
Upload a video and run frame-wise detection
Visualize bounding boxes and confidence scores
Perform inference using the trained YOLOv8 model

▶️ How to Run Locally
1️⃣ Create & Activate Environment
conda create -n yolov8 python=3.10 -y
conda activate yolov8

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

🖼️ Image Inference Example
Upload any road image containing potholes
The model detects potholes and draws bounding boxes in real time

🎥 Video Inference Example
Upload a road video (.mp4)
The system processes each frame
Detected potholes are highlighted throughout the video

🛠️ Tech Stack
Python 3.10
YOLOv8 (Ultralytics)
PyTorch
OpenCV
Streamlit
CUDA (GPU acceleration)

🔒 Model Security Note
The trained model (best.pt) is not included in public repositories to prevent unauthorized reuse.
For deployment, the model can be securely loaded from private storage or private repositories.

📦 Dataset

The dataset used for training is not included in this repository due to size and licensing constraints.
Dataset characteristics:
- Custom pothole detection dataset
- Bounding box annotations
- Train / Validation / Test split

You may use:
- Roboflow pothole datasets
- Custom-collected road images
- Publicly available road damage datasets
Ensure `data.yaml` paths are updated accordingly.

📈 Future Enhancements
Real-time webcam detection
Pothole severity classification
GPS tagging & reporting
ONNX / TensorRT optimization
REST API deployment (FastAPI)

👨‍💻 Author
Rushikesh Vilas Kadam
Machine Learning & GenAI Engineer
📍 India, Pune

⭐ Acknowledgements
Ultralytics YOLOv8
Open-source computer vision community
