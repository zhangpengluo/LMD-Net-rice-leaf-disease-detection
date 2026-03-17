# LMD-Net: A Lightweight Multi-scale Detection Network for Rice Disease Detection in Complex Field Environments

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.1-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Base-YOLOv13n-red.svg" alt="YOLOv13">
  <img src="https://img.shields.io/badge/Params-2.19M-brightgreen.svg" alt="Params">
  <img src="https://img.shields.io/badge/mAP-86.7%25-orange.svg" alt="mAP">
  <img src="https://img.shields.io/badge/Weight-4.9MB-blueviolet.svg" alt="Weight">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <img src="figures/LMD-Net_architecture.jpg" width="90%" alt="LMD-Net Architecture">
</p>

> **LMD-Net** is a lightweight yet powerful object detection network for real-time rice disease detection in complex paddy field environments. With only **2.19M parameters** and **4.9 MB** model weight, it achieves **86.7% mAP** across 6 disease categories — making it ideal for deployment on UAVs, edge devices, and field monitoring terminals.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What's New](#-whats-new)
- [Architecture](#️-architecture)
- [Dataset](#-dataset)
- [Main Results](#-main-results)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference & Validation](#-inference--validation)
- [Visualization](#-visualization)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 📌 Overview

Rice is the staple food for half the world's population, yet diseases such as rice blast, bacterial blight, and brown spot cause severe yield losses annually. Traditional manual diagnosis is inefficient and subjective. **LMD-Net** addresses these challenges by providing a lightweight, accurate, and deployable deep learning solution for automated rice disease detection under complex field conditions including varying illumination, leaf occlusion, and diverse disease morphologies.

### 🎯 Core Challenges Addressed

| Challenge | Solution in LMD-Net |
|-----------|---------------------|
| 🔍 Small-scale disease spots (early-stage lesions) | **LAE** module preserves fine-grained spatial info during downsampling |
| 📐 Multi-scale & irregular lesion morphology | **C3k2_MSCB1** captures features at multiple receptive fields simultaneously |
| 🌿 Complex background interference (occlusion, lighting) | **Detect_MBConv** with SE attention suppresses background noise |
| ⚡ Edge device deployment (UAVs, IoT terminals) | Overall lightweight design: **2.19M params, 5.7 GFLOPs, 4.9 MB** |

---

## 🆕 What's New

- ✅ **LAE (Lightweight Adaptive Extraction)** — A plug-and-play downsampling module using parallel branches + group convolution. Reduces parameters to 1/N of standard convolution while encoding spatial info into channel dimensions via 5D feature representation.
- ✅ **C3k2_MSCB1 (Multi-scale Convolution Block)** — Integrates MSDC (Multi-scale Depth-wise Convolution) with channel shuffle into the C2f cross-stage partial structure for rich multi-scale feature extraction at low computational cost.
- ✅ **Detect_MBConv** — An efficient detection head combining inverted bottleneck (MBConv) + SE channel attention + shortcut connections for enhanced feature discrimination in the detection stage.
- ✅ **Large-scale Rice Disease Dataset** — A custom 6-class dataset with **18,257 original images**, augmented to **~24,785 training images**, with train/val/test three-way split for rigorous evaluation.

---

## 🏗️ Architecture

LMD-Net is built upon **YOLOv13n** with targeted improvements in three key components:

<p align="center">
  <img src="figures/LMD-Net_architecture.jpg" width="90%" alt="LMD-Net Overview">
</p>

### Module-level Modifications

| Location | Original (YOLOv13) | Replacement (LMD-Net) | 
|----------|--------------------|-----------------------|
| **Backbone Downsampling** | Conv | **LAE** | 
| **Backbone & Neck** | DSC3k2 | **C3k2_MSCB1** | 
| **Detection Head** | Detect | **Detect_MBConv** |

### LAE Module

<p align="center">
  <img src="figures/LAE.jpg" width="60%" alt="LAE Module">
</p>

The LAE module consists of two parallel paths:
- **Adaptive Extraction Path**: Average pooling → Conv → Softmax-weighted recombination for adaptive downsampling
- **Lightweight Extraction Path**: 1×1 Conv → Average pooling → Feature rearrangement for efficient dimension mapping

Both paths are fused via weighted summation along the sampling factor dimension `n`, implicitly introducing global context during resolution reduction. Each LAE unit achieves 4× downsampling (H/2 × W/2) while encoding spatial structure into the channel dimension.

### MSCB Module

<p align="center">
  <img src="figures/MSCB.jpg" width="60%" alt="MSCB Module">
</p>

The MSCB module pipeline:

```
Input → 1×1 Conv (channel expansion, C*2)
      → BN + ReLU6
      → MSDC (parallel DW Conv with kernels p×p, q×q, s×s)
      → Channel Shuffle
      → 1×1 Conv (channel projection)
      → BN
      → Residual Add
      → Output
```

MSDC decomposes standard convolution into depth-wise and point-wise stages, using multiple kernel sizes in parallel to capture features at different receptive fields simultaneously.

### MBConv Detection Head

<p align="center">
  <img src="figures/MBConv.jpg" width="40%" alt="MBConv Module">
</p>

```
Input → 1×1 Conv (expand) → 3×3 DW Conv → SE Attention → 1×1 Conv (project) → Shortcut Add → Output
```

When input and output dimensions match, the shortcut connection enables feature reuse and alleviates gradient vanishing in deep networks.

---

## 📊 Dataset

### Overview

We constructed a comprehensive rice disease dataset from two sources:
- 🏫 **Field collection**: Key Laboratory of Smart Agriculture Technology, Ministry of Agriculture and Rural Affairs, Northeast China
- 🌐 **Internet collection**: Publicly available rice disease images with diverse conditions

<p align="center">
  <img src="figures/dataset_construction.jpg" width="85%" alt="Dataset Construction Pipeline">
</p>

### Disease Categories (6 classes, 18,257 original images)

| # | Disease | Chinese Name | Original Images |
|---|---------|-------------|:---------------:|
| 1 | Rice Blast | 稻瘟病 | 3,172 |
| 2 | Bacterial Blight | 白叶枯病 | 3,026 |
| 3 | Brown Spot | 褐斑病 | 3,825 |
| 4 | Rice Tungro | 东格鲁病 | 2,903 |
| 5 | Rice False Smut | 稻曲病 | 787 |
| 6 | Leaf Scald | 叶烧病 | 357 |
| | **Total** | | **18,257** |

### Data Split & Augmentation (Three-way Split)

```
Original 18,257 images
    │
    ├── 70% → SET1 (13,908 images) ──→ Data Augmentation (×2) ──→ Training Set (~24,785 images)
    │         ├── Random horizontal flip
    │         ├── Flip 90° / 180°
    │         ├── Translation
    │         ├── Rotation (small angle)
    │         ├── Brightness & Contrast adjustment
    │         ├── Gaussian noise
    │         └── Motion blur
    │
    ├── 20% → SET2 (2,965 images) ──→ Validation Set (original, no augmentation)
    │
    └── 10% → SET3 (1,384 images) ──→ Test Set (original, no augmentation)
```

> ⚠️ **No data leakage**: Augmentation is applied **only** to the training set. Validation and test sets use original images exclusively.

- **Annotation Format**: YOLO format (`.txt`), each image contains at least one disease category
- **Collection conditions**: Multiple lighting (front-lit / back-lit), shooting angles, and distances to simulate real-world UAV/field robot operating environments

---


## 🚀 Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/zhangpengluo/LMD-Net-rice-leaf-disease-detection.git
cd LMD-Net-rice-leaf-disease-detection

# Create conda environment (recommended)
conda create -n lmdnet python=3.10 -y
conda activate lmdnet

# Install dependencies
pip install -r requirements.txt
```

### System Requirements

| Item | Minimum Specification |
|------|-----------------------|
| OS | Windows 11 / Ubuntu 20.04+ |
| CPU | Intel Core i7-13650HX (or equivalent) |
| GPU | NVIDIA RTX 4060 Laptop 8GB (or better) |
| Python | 3.10.19 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |

---

## 🏋️ Training

### Dataset Preparation

Organize your dataset in standard YOLO format with **three splits** (train/val/test):

```
datasets/
└── rice_disease/
    ├── images/
    │   ├── train/          # Training images (original + augmented, ~24,785)
    │   │   ├── img_0001.jpg
    │   │   └── ...
    │   ├── val/            # Validation images (original only, 2,965)
    │   │   ├── img_0001.jpg
    │   │   └── ...
    │   └── test/           # Test images (original only, 1,384)
    │       ├── img_0001.jpg
    │       └── ...
    ├── labels/
    │   ├── train/          # YOLO format .txt annotations
    │   │   ├── img_0001.txt
    │   │   └── ...
    │   ├── val/
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── data.yaml           # Dataset config file
```

**data.yaml** example:

```yaml
path: datasets/rice_disease
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: RiceBlast
  1: BacterialBlight
  2: BrownSpot
  3: RiceTungro
  4: RiceFalseSmut
  5: LeafScald
```

### Start Training
**bash**
```
python train.py \
  --data datasets/rice_disease/data.yaml \
  --cfg models/LMD-Net.yaml \
  --epochs 400 \
  --batch 8 \
  --imgsz 320 \
  --optimizer SGD \
  --patience 100 \
  --device 0 \
  --workers 4 \
  --seed 0 \
  --amp
```
**python**
```
from ultralytics import YOLO, RTDETR
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    model = YOLO(r'F:\rice leaf disease\yolov13-main\ultralytics\cfg\models\v13\MSCB1_LAE_DetectMBConv.yaml') #change into your model path
		#model = RTDETR(r'F:\rice leaf disease\yolov13-main\ultralytics\cfg\models\v13\MSCB1_LAE_DetectMBConv.yaml')#if use RTDETR
    results = model.train(
        data=r'F:\rice leaf disease\Rice Tungro\Tungro.yaml',#change into your data path
        epochs=250,
        batch=16,
        imgsz=640,
        device=0,
        patience=60,
        workers=0, 
        cache='ram',  
        amp=True, 
        project="runs",
        name="Rice Tungro",
        optimizer='SGD',
        resume=True,
    )
    metrics = model.val()
``` 

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 400 | Maximum training epochs |
| `patience` | 100 | Early stopping patience |
| `batch` | 8 | Batch size per GPU |
| `imgsz` | 320 | Input image size |
| `optimizer` | SGD | Optimizer type |
| `seed` | 0 | Random seed for reproducibility |
| `amp` | True | Automatic Mixed Precision training |
| `mosaic` | 1.0 | Mosaic augmentation ratio |
| `device` | 0 | GPU device index |
| `workers` | 4 | Data loading workers |

---

## 🔍 Inference & Validation

### Single Image / Folder Inference
**bash**
```
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source path/to/images \
  --imgsz 320 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --device 0
```
**python**
```
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR
if __name__ == '__main__':
    model = YOLO(r'F:\rice leaf disease\yolov13-main\runs\Leaf_Scald\weights\best.pt') # select your model.pt path
    #model = RTDETR(r"F:\rice leaf disease\yolov13-main\rtdetr-x\train\weights\best.pt")
    model.predict(source=r'F:\rice leaf disease\Leaf_Scald\train\images\aug_0_7_jpg.rf.6193c60a84c4a39d22e3bb0d3be8b816.jpg',
                  imgsz=1280,
                  project='runs/detect',
                  name='Scald',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
```

### Video Inference

```bash
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source path/to/video.mp4 \
  --imgsz 320 \
  --device 0
```

### Validation (on val set)

```bash
python val.py \
  --data datasets/rice_disease/data.yaml \
  --weights runs/train/exp/weights/best.pt \
  --imgsz 320 \
  --batch 8 \
  --device 0
```



## 🔥 Visualization

### Detection Results Comparison

<p align="center">
  <img src="figures/detection_comparison.jpg" width="95%" alt="Detection Results Comparison">
</p>

> Visual comparison across 6 disease types with YOLOv7n/tiny, YOLOv8n, YOLOv11n, YOLOv13n (baseline), RT-DETR series, and **LMD-Net (Ours)**. LMD-Net shows superior performance especially on:
> - 🔬 **Small-scale lesions** (early rice blast spots)
> - 📦 **Densely distributed disease spots** (brown spot clusters)
> - 🎯 **Irregular morphology** (bacterial blight elongated lesions)

### Grad-CAM Heatmap Analysis

<p align="center">
  <img src="figures/gradcam_heatmap.jpg" width="95%" alt="Grad-CAM Heatmap Analysis">
</p>

> Progressive module integration analysis using Grad-CAM visualization across all 6 disease categories:

| Configuration | Attention Pattern |
|---------------|-------------------|
| **YOLOv13n (Baseline)** | Diffused, easily distracted by leaf edges and background |
| **+ LAE only** | Begins converging toward lesion areas, overcomes background interference |
| **+ MSCB1 only** | Better coverage of multi-scale disease spots |
| **+ MBConv only** | Sharper boundaries, more focused on core necrotic regions |
| **LMD-Net (All three)** | **Maximum focus** on true pathological features with minimal background activation |

> The synergy of LAE + MSCB1 + MBConv achieves the most concentrated energy distribution on disease-critical features, explaining why LMD-Net outperforms the baseline despite having fewer parameters.


## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{zhang2025lmdnet,
  title={LMD-Net: A Lightweight Multi-scale Detection Network for Rice Disease Detection in Complex Field Environments},
  author={Zhang, Pengluo},
  journal={},
  year={2025},
  url={https://github.com/zhangpengluo/LMD-Net-rice-leaf-disease-detection}
}
```

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO ecosystem and training framework
- [YOLOv13](https://github.com/ultralytics/ultralytics) — Baseline detection model
- **Key Laboratory of Smart Agriculture Technology**, Ministry of Agriculture and Rural Affairs, Northeast China — Field data collection and experimental support

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 📧 Contact

If you have any questions or suggestions, please open an [Issue](https://github.com/zhangpengluo/LMD-Net-rice-leaf-disease-detection/issues) or contact:

- **GitHub**: [@zhangpengluo](https://github.com/zhangpengluo)

---

<p align="center">
  ⭐ If you find this project helpful, please give it a star! ⭐
</p>
