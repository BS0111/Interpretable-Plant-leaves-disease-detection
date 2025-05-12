# 🌿 An Interpretable Plant Leaf Disease Detection using Convolution Block Attention Mechanism

This repository contains a deep learning pipeline for classifying plant leaf diseases using a modified VGG16 model enhanced with CBAM (Convolutional Block Attention Module). It also features explainability tools such as Grad-CAM, Grad-CAM++, and Layer-wise Relevance Propagation (LRP) to interpret model predictions.

---

## 🧾 Table of Contents

- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Visualization](#-visualization)
- [LRP Explanations](#-lrp-explanations)
- [Evaluation](#-evaluation)
- [Dependencies](#-dependencies)
- [Notes](#-notes)
- [Author](#-author)
- [Contact](#-contact)

---

## 🗂️ Project Structure

```text
project/
├── test_images/                 # Sample test images
├── attributions/               # Generated heatmaps (CBAM, GradCAM, etc.)
├── cbam_module.py              # CBAM module implementation
├── vgg16_cbam_model.py         # VGG16 model with CBAM
├── train_cbam_vgg16.py         # Training script
├── visualize_cbam_overlay.py   # Visualize CBAM attention on images
├── gradcam_utils.py            # Grad-CAM and Grad-CAM++ logic
├── run_gradcam.py              # Script for Grad-CAM and Grad-CAM++ visualizations
├── run_lrp.py                  # LRP visualization
├── utils.py                    # Utility functions
└── README.md
```

---

## 🔍 Key Features

- ✅ CBAM-enhanced VGG16 for attention-aware learning  
- ✅ Grad-CAM and Grad-CAM++ for post-hoc explanation  
- ✅ LRP visualizations 
- ✅ Well-structured and reproducible codebase  

---

## 🧠 Model Architecture

- **Base**: VGG16 pretrained on ImageNet  
- **Modification**: CBAM block added after each convolutional block  
- **Output**: Softmax classification over plant disease categories  

---

## 🚀 Training

Train the CBAM-VGG16 model:

```bash
python train_cbam_vgg16.py --data_dir /path/to/train --epochs 20 --batch_size 32
```

---

## 📸 Visualization

### CBAM Overlay:

```bash
python visualize_cbam_overlay.py --input_dir test_images --output_dir attributions --model_path path/to/model.pth
```

### Grad-CAM:

```bash
python run_gradcam.py --input_dir test_images --model_path path/to/model.pth --technique gradcam
```

### Grad-CAM++:

```bash
python run_gradcam.py --input_dir test_images --model_path path/to/model.pth --technique gradcam++
```

---

## 🔬 LRP Explanations

Run LRP:

```bash
python run_lrp.py --input_dir test_images --model_path path/to/model.pth --rule epsilon_alpha2_beta1
```

---

## 📦 Dependencies

```text
python==3.10
torch==2.1.2
torchvision==0.16.2
numpy==1.26.3
matplotlib==3.8.0
opencv-python==4.9.0.80
zennit==0.5.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Balram Singh**  
National Institute of Technology Hamirpur

---

## 📧 Contact

For queries or collaborations: `23mcs104@nith.ac.in`
