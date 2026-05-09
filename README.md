# MultiTask-ViT-AV2
Multi-task learning for vehicle intention and trajectory prediction from LiDAR using Vision Transformers — GUC Bachelor Thesis
 
Bachelor Thesis — Faculty of Media Engineering and Technology (MET)  
German University in Cairo (GUC)  
Author: [Your Name]  
Supervisors: Dr. Milad Michel Ghantous, Eng. Noha Hamid  

---

## Overview

This repository implements a multi-task learning framework for jointly 
predicting vehicle detections, intentions (8-class), and trajectories 
(6-mode multimodal) from Bird's-Eye-View LiDAR representations on the 
Argoverse 2 Sensor dataset.

Three model variants are implemented and compared:

| Model | Backbone | Trajectory Head | Intention |
|-------|----------|-----------------|-----------|
| V1 — Baseline | ViT-Small patch8 (scratch) | None | 8-class head |
| V2 — MT-MLP | ViT-Small patch8 (scratch) | MLP decoder | Derived from trajectory |
| V3 — MT-Swin | Swin-Tiny (ImageNet pretrained) | Transformer decoder | Derived from trajectory |

This work extends:
- [IntentNetViT](https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction) 
  by Nadeem Mohamed (intention prediction)
- [HiVT-AV2](https://github.com/mohamed-abdulbaki22/HiVT-av2) 
  by Mohamed Abdulbaki (trajectory prediction)

---

## Repository Structure

IntentTrajNet-AV2/
├── models/               # Model architecture
│   ├── backbone.py       # ViT and Swin backbones
│   ├── heads.py          # Detection, Intention, Trajectory heads
│   ├── trajectory_decoder.py  # MLP and Transformer decoders
│   └── model_mt.py       # Full multi-task model
├── training/             # Training and evaluation
│   ├── loss.py           # All loss functions
│   ├── train.py          # Unified trainer
│   └── eval.py           # Evaluation with all metrics
├── datasets/             # Data loading
│   └── av2_dataset.py    # AV2 dataset with trajectory GT
├── configs/              # Model configurations
│   ├── v1_baseline.yaml
│   ├── v2_mlp.yaml
│   └── v3_swin.yaml
├── utils/                # Shared utilities
├── data/                 # Data processing scripts
│   └── sensor_to_mf.py   # Sensor-to-MF conversion pipeline
└── inspection/           # Analysis and inspection scripts

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Train V1 baseline
python training/train.py --config configs/v1_baseline.yaml

# Train V2 multi-task MLP
python training/train.py --config configs/v2_mlp.yaml

# Train V3 multi-task Swin
python training/train.py --config configs/v3_swin.yaml
```

---

## Evaluation

```bash
python training/eval.py --config configs/v2_mlp.yaml \
  --checkpoint checkpoints/v2_best.pth
```