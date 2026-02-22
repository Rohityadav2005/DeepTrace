# **DeepTrace – Deepfake Detection System**

## 📌 Project Overview

**DeepTrace** is an AI-powered deepfake detection system designed to identify manipulated videos by analyzing facial features and temporal inconsistencies. The project leverages **deep learning**, combining **CNN (ResNet)** for spatial feature extraction and **LSTM** for temporal sequence learning.

This system helps in detecting synthetic media, improving digital trust, and combating misinformation.

---

## 🎯 Objectives

* Detect whether a video is **Real** or **Fake**
* Extract and analyze facial frames from videos
* Learn spatial + temporal patterns of deepfakes
* Provide an automated pipeline for deepfake detection

---

## 🧠 Tech Stack

### **Languages**

* Python

### **Libraries & Frameworks**

* PyTorch
* OpenCV
* NumPy
* RetinaFace / YOLO-Face (for face detection)
* torchvision

### **Deep Learning Models**

* **ResNet** → Feature extraction from frames
* **LSTM** → Temporal sequence learning

---

## ⚙️ System Architecture

### **Step 1: Video Input**

* Input videos are categorized into:

  * Real videos folder
  * Fake videos folder

### **Step 2: Face Detection**

* Faces are extracted frame-by-frame using:

  * RetinaFace OR YOLOv5-Face

### **Step 3: Preprocessing**

* Crop faces
* Resize images
* Normalize pixel values
* Convert BGR → RGB

### **Step 4: Feature Extraction**

* ResNet extracts spatial features from each frame

### **Step 5: Temporal Learning**

* LSTM processes frame sequences to learn motion patterns

### **Step 6: Classification**

* Final output:

  * **Real**
  * **Fake**

---


## 🚀 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Rohityadav2005/DeepTrace.git
cd deeptrace
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### **For Detecting the face**

```bash
python detecting_face2.py
```

## 📊 Dataset

* Deepfake video datasets containing:

  * Real videos (kaggle dataset Deepfake)
  * Manipulated videos



* FaceForensics++

---

## 🔥 Key Features

✔ Automated face extraction pipeline
✔ Hybrid CNN + LSTM architecture
✔ Temporal deepfake detection
✔ Scalable training workflow
✔ High accuracy on benchmark datasets

---

## 📈 Future Improvements

* Real-time deepfake detection
* Web interface deployment
* Transformer-based architecture
* Multi-face detection support
* Explainable AI visualization

---

## 👨‍💻 Author

**Rohit Yadav Ji**

---

## 📜 License

This project is for educational and research purposes.

---

## ⭐ Acknowledgment

Thanks to the open-source deep learning community and datasets that made this research possible.

---

## 💡 Tagline

> **DeepTrace – Tracing the Truth Behind Every Frame**
