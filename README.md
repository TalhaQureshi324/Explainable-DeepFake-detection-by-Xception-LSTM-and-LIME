# 🎭 DeepFake Detection Using Spatiotemporal Deep Learning with Explainable AI

> **Final Year Project (FYP)**  
> **Domain:** Computer Vision & Deep Learning  
> **Deployment:** [Hugging Face Spaces – Deep-Fake-Detection](https://huggingface.co/spaces/TalhaQureshi324/Deep-Fake-Detection) *(Live Demo – You can verify its functioning here)*

---

## 📋 Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Literature & Motivation](#literature--motivation)
5. [System Architecture](#system-architecture)
6. [Dataset](#dataset)
7. [Methodology](#methodology)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Model Architecture (V1 & V2)](#model-architecture-v1--v2)
   - [Training Strategy](#training-strategy)
   - [Ensemble Prediction (V2)](#ensemble-prediction-v2)
   - [Explainable AI (XAI)](#explainable-ai-xai)
8. [Project Structure](#project-structure)
9. [Installation & Setup](#installation--setup)
10. [Usage Guide](#usage-guide)
    - [Training](#training)
    - [Testing & Inference](#testing--inference)
    - [Validation](#validation)
    - [Deployment (Gradio App)](#deployment-gradio-app)
11. [Results & Evaluation](#results--evaluation)
12. [Key Features](#key-features)
13. [Future Work](#future-work)
14. [Team & Acknowledgements](#team--acknowledgements)
15. [References](#references)

---

## Abstract

With the rapid advancement of Generative Adversarial Networks (GANs) and diffusion models, synthetic media—commonly known as **DeepFakes**—have reached a level of realism that makes manual detection nearly impossible. DeepFakes pose severe threats to personal privacy, political integrity, financial security, and societal trust. This project proposes a **robust, automated DeepFake detection framework** that combines **spatiotemporal deep learning** with **Explainable AI (XAI)** to classify video and image content as **REAL** or **FAKE**.

Our approach leverages a **hybrid CNN–RNN architecture**: **Xception**, pre-trained on ImageNet, serves as the spatial feature extractor for individual frames, while a **Long Short-Term Memory (LSTM)** network models temporal inconsistencies across video frames. Two model variants were developed—**Version 1 (10-frame)** and **Version 2 (32-frame with ensemble prediction)**—with the latter significantly improving detection robustness through sliding-window aggregation. Furthermore, we integrate **LIME (Local Interpretable Model-agnostic Explanations)** to generate visual heatmaps, making the model's decisions transparent and interpretable for end-users.

The final system is deployed as a **bilingual (English / Urdu) Gradio web application** on **Hugging Face Spaces**, enabling real-world accessibility for users across diverse demographics.

---

## Problem Statement

DeepFake technology has democratized the creation of hyper-realistic fake videos, lowering the barrier for misinformation, identity theft, cyberbullying, and political manipulation. Existing detection methods suffer from one or more of the following limitations:

- **Poor generalization** to unseen forgery techniques.
- **Lack of temporal modeling**, focusing only on single-frame artifacts.
- **Black-box nature**, offering no explanation for decisions.
- **No accessible deployment**, limiting real-world usability.

This project addresses all four gaps by designing a **generalizable, temporal, explainable, and deployable** detection system.

---

## Objectives

1. **Develop an end-to-end deep learning pipeline** for DeepFake video/image detection.
2. **Extract spatiotemporal features** using a CNN–LSTM hybrid architecture.
3. **Improve detection robustness** via ensemble prediction and class balancing.
4. **Provide interpretability** through LIME-based XAI heatmaps.
5. **Deploy a user-friendly web application** with multilingual support (English & Urdu).

---

## Literature & Motivation

Traditional DeepFake detection relied on hand-crafted features (e.g., eye-blinking analysis, face-warping artifacts). However, as generators improved, these heuristics failed. Modern approaches fall into three categories:

| Category | Approach | Limitation |
|----------|----------|------------|
| **Spatial** | Frame-level CNN classifiers | Ignores temporal coherence |
| **Temporal** | Optical flow, LSTM, 3D-CNN | High compute, complex training |
| **Hybrid** | CNN + RNN / Transformer | Best balance; adopted in this project |

Our hybrid **Xception + LSTM** design is motivated by:
- **Xception's** superior performance on fine-grained visual tasks due to depthwise separable convolutions.
- **LSTM's** ability to capture long-range temporal dependencies without vanishing gradients.
- The proven effectiveness of transfer learning from ImageNet to face-related tasks.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEEPFAKE DETECTION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐      │
│   │   Input     │───▶│  Preprocessing   │───▶│  Face Extraction     │      │
│   │ (Video/Img) │    │  (OpenCV DNN)    │    │  + BG Removal        │      │
│   └─────────────┘    └──────────────────┘    └──────────────────────┘      │
│                                                       │                     │
│                                                       ▼                     │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    SPATIOTEMPORAL MODEL                         │      │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │      │
│   │  │   Frames    │───▶│  Xception   │───▶│  Global AvgPool │     │      │
│   │  │ 299×299×3   │    │  (CNN)      │    │  (per frame)    │     │      │
│   │  └─────────────┘    └─────────────┘    └─────────────────┘     │      │
│   │         │                                          │            │      │
│   │         ▼                                          ▼            │      │
│   │  ┌─────────────────────────────────────────────────────────┐    │      │
│   │  │              LSTM (64 units)                            │    │      │
│   │  │         Temporal feature aggregation                    │    │      │
│   │  └─────────────────────────────────────────────────────────┘    │      │
│   │                              │                                  │      │
│   │                              ▼                                  │      │
│   │  ┌─────────────────────────────────────────────────────────┐    │      │
│   │  │              Dense (1) + Sigmoid                        │    │      │
│   │  │         Fake Probability [0, 1]                         │    │      │
│   │  └─────────────────────────────────────────────────────────┘    │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                        POST-PROCESSING                            │    │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │    │
│   │  │   Ensemble  │───▶│  Threshold  │───▶│  LIME Explanation   │   │    │
│   │  │  (V2 only)  │    │   (0.75)    │    │  (XAI Heatmap)      │   │    │
│   │  └─────────────┘    └─────────────┘    └─────────────────────┘   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         GRADIO UI                                   │  │
│   │     Bilingual (EN/UR) · Light/Dark Theme · Video + Image Support    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

This project uses the **FaceForensics++ (FF++)** dataset, one of the most widely adopted benchmarks for DeepFake detection research.

### Dataset Organization

```
data/
├── train/
│   ├── REAL/          # Original (unmanipulated) videos
│   └── FAKE/          # DeepFake-manipulated videos
└── val/
    ├── REAL/
    └── FAKE/
```

### Data Statistics

| Split | REAL Videos | FAKE Videos | Total |
|-------|-------------|-------------|-------|
| Train | ~70% | ~70% | Majority of data |
| Val   | ~30% | ~30% | Held-out for evaluation |

> **Note:** A custom `script.py` was used to reorganize the raw FaceForensics++ directory structure (`DFD_original_sequences` and `DFD_manipulated_sequences`) into the above `train/val` split with `REAL/FAKE` categories.

### Preprocessing Output

After preprocessing, each video is converted into a folder of extracted face frames:

```
preprocessed_new/
├── train/
│   ├── REAL/
│   │   └── video_name/
│   │       ├── frame_0000_face_0.jpg
│   │       ├── frame_0001_face_0.jpg
│   │       └── ...
│   └── FAKE/
└── val/
    ├── REAL/
    └── FAKE/
```

---

## Methodology

### Preprocessing Pipeline

The preprocessing stage is critical because the model learns from **face regions only**. Background clutter introduces noise and hurts generalization.

#### Step 1: Face Detection
- **Tool:** OpenCV DNN Face Detector (`deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`)
- **Confidence Threshold:** 0.5
- Returns bounding box `(x1, y1, x2, y2)` for the most confident face per frame.

#### Step 2: Face Extraction with Margin
- A **20% margin** is added around the detected bounding box to include contextual facial regions (ears, hairline, jawline) that often contain GAN artifacts.

#### Step 3: Background Removal
- An **elliptical mask** is applied to black out everything outside the face oval.
- The mask is **Gaussian-blurred** at the edges for smooth feathering, preventing hard-edge artifacts.
- This forces the model to focus exclusively on facial features.

#### Step 4: Resizing & Normalization
- All faces are resized to **299×299** (Xception input size).
- Pixel values are normalized to **[0, 1]** during training/inference.

#### Step 5: Frame Sampling
- For videos, **evenly spaced frames** are extracted (10 for V1, 32 for V2).
- If a video has fewer frames than required, the last frame is **replicated (padded)** to maintain fixed input dimensions.

---

### Model Architecture (V1 & V2)

Both versions share the same base architecture but differ in temporal depth and prediction strategy.

```python
# Shared Backbone Architecture
Input: (NUM_FRAMES, 299, 299, 3)
    │
    ├──▶ TimeDistributed(Xception weights='imagenet', trainable=True)
    │         └── Fine-tuning: first 100 layers frozen
    ├──▶ TimeDistributed(GlobalAveragePooling2D())
    ├──▶ LSTM(64)
    └──▶ Dense(1, activation='sigmoid')
```

| Component | Details |
|-----------|---------|
| **CNN Backbone** | Xception (ImageNet pre-trained) |
| **Frozen Layers** | First 100 layers (low-level features) |
| **Temporal Model** | LSTM with 64 hidden units |
| **Output** | Sigmoid (Fake probability) |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam (lr = 1e-5) |

#### Version 1 (`deepfake_detector_model.keras`)
- **Input:** 10 frames per video
- **Batch Size:** 2
- **Training Epochs:** 10
- **Best Model Saved:** Based on `val_loss` (min)
- **Class Balancing:** Computed class weights (`total / (n_classes * count)`) to handle imbalance without discarding data.

#### Version 2 (`deepfake_detector_model_v2.keras`)
- **Input:** 32 frames per video
- **Minimum Frames:** 5 (shorter clips accepted)
- **Batch Size:** 1 (memory constraint due to 3× more frames)
- **Key Improvement:** **Ensemble Prediction** (see below)

---

### Ensemble Prediction (V2)

Version 2 introduces a **multi-window ensemble** strategy that significantly improves robustness:

1. **Full-Sequence Prediction:** The entire 32-frame sequence is classified once.
2. **Sliding-Window Predictions:** 10-frame windows slide across the video with a stride of 5 (50% overlap). Each window is independently padded to 32 frames and classified.
3. **Aggregation:** The **mean** of all window predictions is taken as the final score.

```
Video Frames: [f0][f1][f2]...[f31]

Full-32  : [================================]  → Prediction 1
Win-0-9  : [==========]....................  → Prediction 2
Win-5-14 : .....[==========]...............  → Prediction 3
Win-10-19: ..........[==========]..........  → Prediction 4
   ...

Final Score = mean(Prediction_1, Prediction_2, ...)
```

This ensemble approach:
- **Reduces variance** caused by anomalous single frames.
- **Captures local temporal artifacts** that may be missed by global averaging.
- **Improves confidence calibration** via standard deviation reporting.

---

### Explainable AI (XAI)

Deep learning models are often criticized as "black boxes." To address this, we integrate **LIME (Local Interpretable Model-agnostic Explanations)**:

1. The **middle frame** of the sequence is selected as the representative image.
2. LIME generates perturbed versions of this frame (superpixel masking).
3. The model predicts on each perturbed version.
4. LIME fits an interpretable linear model to identify **which superpixels most influence the Fake/Real decision**.
5. A **heatmap overlay** is generated, highlighting suspicious facial regions (e.g., unnatural skin texture, misaligned eyes, warped boundaries).

> **XAI Output:** `xai_outputs/xai_<video_name>.png`

---

## Project Structure

```
deepfake-main/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
│
├── 🤖 TRAINING
│   ├── main.py                           # V1 Training Script (10-frame model)
│   ├── main_version2.py                  # V2 Training Script (32-frame + ensemble)
│   ├── val.py                            # Model evaluation on validation set
│   └── preprocess_all.py                 # Legacy Haar Cascade preprocessing
│
├── 🧪 TESTING & INFERENCE
│   ├── test.py                           # V1 Testing + LIME XAI
│   ├── test_version2.py                  # V2 Testing + Ensemble + LIME XAI
│   ├── explain.py                        # Standalone LIME explanation module
│   └── testing/                          # Folder for test videos
│       ├── ASSAD.mp4
│       ├── mine2.mp4
│       └── shaheer boi.mp4
│
├── 🖼️ PREPROCESSING
│   ├── preprocessing.py                  # Main preprocessing module (OpenCV DNN)
│   ├── preprocess.py                     # Alternative preprocessing (MediaPipe)
│   └── preprocessed_new/                 # Output directory for extracted faces
│
├── 🚀 DEPLOYMENT
│   ├── app.py                            # Gradio web app (Hugging Face Spaces)
│   ├── deepfake_detector_model.keras     # V1 trained model (~166 MB)
│   └── deepfake_detector_model_v2.keras  # V2 trained model (~166 MB)
│
├── 🗂️ DATASET ORGANIZATION
│   ├── script.py                         # Reorganizes FF++ into train/val splits
│   └── data/                             # Raw video dataset
│       ├── train/
│       │   ├── REAL/
│       │   └── FAKE/
│       └── val/
│           ├── REAL/
│           └── FAKE/
│
├── 📊 XAI OUTPUTS
│   ├── xai_outputs/                      # V1 LIME heatmaps
│   ├── xai_outputs_v2/                   # V2 LIME heatmaps
│   ├── temp_runtime_frames/              # V1 temp extraction cache
│   ├── temp_runtime_frames_v2/           # V2 temp extraction cache
│   └── xai_*.png                         # Sample XAI visualizations
│
└── 🖼️ SAMPLE IMAGES
    ├── case1.jpeg
    ├── case2.png
    └── xai_mine.png
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for training; CPU works for inference)
- Git LFS (for model files)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd deepfake-main
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | ≥2.15.0 | Deep learning framework |
| OpenCV (headless) | ≥4.8.0 | Face detection & image processing |
| NumPy | ≥1.24.0 | Numerical computation |
| Pillow | ≥10.0.0 | Image I/O |
| Gradio | ≥4.0.0 | Web UI deployment |
| LIME | ≥0.2.0 | XAI explanations |
| scikit-image | ≥0.22.0 | Segmentation for LIME |
| matplotlib | latest | Visualization |
| tqdm | latest | Progress bars |

### Step 4: Download Face Detector Weights

The OpenCV DNN face detector files are **auto-downloaded** on first run by `app.py`, or you can manually place them in the project root:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

### Step 5: Prepare Dataset

Organize your FaceForensics++ videos as:
```
data/
├── train/REAL/   # Original videos (.mp4)
├── train/FAKE/   # Manipulated videos (.mp4)
├── val/REAL/
└── val/FAKE/
```

Use `script.py` to automate reorganization from raw FF++ downloads.

---

## Usage Guide

### Training

#### Train Version 1 (10-Frame Model)
```bash
python main.py
```
- Processes `preprocessed_new/` directory.
- Saves best model to `deepfake_detector_model.keras`.

#### Train Version 2 (32-Frame Model)
```bash
python main_version2.py
```
- Uses up to 32 frames per video with sequence padding.
- Saves best model to `deepfake_detector_model_v2.keras`.

### Preprocessing

#### Full Dataset Preprocessing
```bash
python preprocessing.py --mode dataset
```

#### Single Video Preprocessing
```bash
python preprocessing.py --mode video -i "path/to/video.mp4" -o "output_folder"
```

#### Single Image Preprocessing
```bash
python preprocessing.py --mode image -i "path/to/image.jpg"
```

### Testing & Inference

#### Test Version 1
```bash
python test.py
```
- Place test videos in `testing/` folder.
- Outputs predictions + XAI heatmaps to `xai_outputs/`.

#### Test Version 2 (Recommended)
```bash
python test_version2.py
```
- Uses **ensemble prediction** for higher accuracy.
- Reports: **Verdict, Average Score, Confidence, Std Deviation, Windows Used**.
- Generates XAI heatmaps in `xai_outputs_v2/`.

#### Standalone XAI Explanation
```python
from explain import explain_prediction
explain_prediction("deepfake_detector_model.keras", processed_frames_array)
```

### Validation

```bash
python val.py
```
- Evaluates the V1 model on the validation set.
- Prints validation loss, accuracy, and sample predictions.

### Deployment (Gradio App)

#### Run Locally
```bash
python app.py
```
- Opens at `http://localhost:7860`
- Supports **video and image** uploads.
- Toggle **English / Urdu** language.
- Toggle **Light / Dark** theme.
- Enable **XAI Explanation** checkbox for LIME heatmaps.

#### Deploy to Hugging Face Spaces
```bash
# Ensure model is tracked with Git LFS
git lfs track "*.keras"
git add .
git commit -m "Deploy DeepFake Detection App"
git push
```

> **🚀 Live Deployment:** [https://huggingface.co/spaces/TalhaQureshi324/Deep-Fake-Detection](https://huggingface.co/spaces/TalhaQureshi324/Deep-Fake-Detection)  
> *You can verify its functioning by uploading any video or image containing a face.*

---

## Results & Evaluation

### Decision Threshold
- **Threshold:** 0.75 (tunable in `app.py` and `test_version2.py`)
- **Score > 0.75:** Classified as **FAKE**
- **Score ≤ 0.75:** Classified as **REAL**

### Metrics Reported (V2)
| Metric | Description |
|--------|-------------|
| **Average Score** | Mean prediction across all ensemble windows |
| **Confidence** | `score` if FAKE, else `1 - score` |
| **Std Deviation** | Lower = model is more certain |
| **Windows Used** | Number of ensemble predictions made |

### Qualitative Results
The XAI heatmaps consistently highlight regions where GAN-based manipulations are most visible:
- **Unnatural skin smoothing** around cheeks and forehead
- **Misaligned facial boundaries** near the jawline
- **Inconsistent eye reflections** and iris details
- **Blurred hair-flesh transitions**

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Dual Model Support** | V1 (10-frame) and V2 (32-frame ensemble) |
| **OpenCV DNN Face Detection** | Robust, no external dependencies beyond OpenCV |
| **Elliptical Background Removal** | Reduces noise, focuses model on facial regions |
| **Sliding-Window Ensemble** | V2 only; improves accuracy and calibration |
| **LIME XAI Explanations** | Transparent, interpretable heatmaps |
| **Bilingual UI** | English & Urdu full interface localization |
| **Theme Switching** | Light & Dark modes with custom CSS |
| **Image + Video Support** | Unified pipeline for both media types |
| **Auto Frame Detection** | `app.py` auto-detects model input shape at runtime |
| **Cloud Deployed** | Hugging Face Spaces with zero-config sharing |

---

## Future Work

1. **Transformer Backbones:** Replace LSTM with Video Swin Transformer or TimeSformer for longer temporal context.
2. **Multi-Face Support:** Extend detection to videos with multiple faces simultaneously.
3. **Audio-Visual Fusion:** Integrate lip-sync inconsistency detection (audio branch).
4. **Adversarial Robustness:** Test and defend against adversarial perturbations.
5. **Mobile Optimization:** Convert to TensorFlow Lite for edge deployment.
6. **Real-Time Streaming:** Optimize pipeline for live webcam feed analysis.

---

## Team & Acknowledgements

This project was developed as a **Final Year Project** in the domain of Computer Vision and Deep Learning.

**Special thanks to:**
- The creators of **FaceForensics++** for the benchmark dataset.
- The **TensorFlow** and **Keras** teams for the deep learning framework.
- **Hugging Face** for providing free Spaces hosting.
- The open-source community behind **OpenCV**, **Gradio**, and **LIME**.

---

## References

1. Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). **FaceForensics++: Learning to Detect Manipulated Facial Images.** *ICCV*.
2. Chollet, F. (2017). **Xception: Deep Learning with Depthwise Separable Convolutions.** *CVPR*.
3. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory.** *Neural Computation*.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). **"Why Should I Trust You?": Explaining the Predictions of Any Classifier.** *KDD*.
5. Dolhansky, B., et al. (2020). **The DeepFake Detection Challenge (DFDC) Dataset.** *arXiv*.

---

> **⚠️ Disclaimer:** This tool is intended for research, educational, and defensive cybersecurity purposes only. Misuse of DeepFake detection technology or the underlying dataset for harmful purposes is strictly condemned.

---

<p align="center">
  <b>🎓 Final Year Project – DeepFake Detection 🎓</b><br>
  <a href="https://huggingface.co/spaces/TalhaQureshi324/Deep-Fake-Detection">🚀 Try the Live Demo</a>
</p>
