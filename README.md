# STAFDD: A Spatio-Temporal Automatic Fish Disease Detection Method

This repository provides the **official implementation** of the paper:

**STAFDD: A Spatio-Temporal Automatic Fish Disease Detection Method**  
*Bo Wang et al.*

---

## 🔍 Overview

STAFDD is a **spatio-temporal automatic fish disease detection framework** designed for
high-density aquaculture environments.  
The framework integrates **object detection, multi-target tracking, and temporal behavior analysis** to assess fish health status from video data.

Core components include:

- **SSCA-YOLO**: an improved YOLOv8-based detector for fish body-surface detection
- **ByTeSort**: a customized multi-target tracking algorithm proposed in this work
- **ReID-based identity association**
- **LSTM-based temporal behavior modeling** for health assessment

---

## 🖥️ Environment

The codebase has been developed and tested under the following environment:

- **OS**: Linux 5.14.0 (Rocky Linux 9.6)
- **Python**: 3.12.9
- **PyTorch**: 2.7.0 + CUDA 12.6
- **CUDA**: 12.6 (Available)
- **cuDNN**: 90501

### Key Dependencies

```text
ultralytics           8.3.127
ultralytics-thop      2.0.14
```
⚠️ Note: Compatibility with other versions is not guaranteed.

📁 Repository Structure

The repository is organized as follows:

```text
STAFDD/
├── README.md
├── LICENSE
├── SSCA-YOLO.yaml
├── tran_yolo.py
├── ByTeSort.py
├── LSTM/
├── docker/
├── docs/
├── track/
├── ReID.pt
└── ultralytics/
```

Core Components

SSCA-YOLO.yaml
Configuration file defining the training framework of the proposed SSCA-YOLO detector, including network structure and training settings.

tran_yolo.py
Script for training the detection model, containing detailed parameter settings and data augmentation strategies used in this study.

ByTeSort.py
The customized multi-target tracking algorithm proposed in this paper, designed to enhance identity association and trajectory stability in dense aquaculture scenes.

LSTM/
Scripts implementing the LSTM-based spatio-temporal behavior analysis module, which analyzes fish trajectories to assess health status.

ReID.pt
Pretrained Re-identification (ReID) model used to maintain identity consistency during multi-target tracking.

Other directories (e.g., docker/, docs/, track/, ultralytics/) provide supporting code and dependencies and are not the primary focus of this work.

📊 Dataset

The YOLO training dataset, pretrained model weights, test videos, and test results are publicly available at:

👉 Hugging Face Dataset
https://huggingface.co/datasets/wangbo66/STAFDD-dataset

The dataset includes:

YOLO-format annotated images

Trained .pt model weights

Raw test videos

Corresponding inference and evaluation results

Please refer to the dataset card on Hugging Face for detailed data organization and usage instructions.

🚀 Usage
Training

Training is performed using the SSCA-YOLO framework defined in SSCA-YOLO.yaml, with parameters specified in tran_yolo.py.

```bash
python tran_yolo.py
```

Tracking and Behavior Analysis

Detection results are processed using the proposed ByTeSort tracker.

Fish trajectories are further analyzed using the LSTM-based scripts in the LSTM/ directory to infer health-related behaviors.

📌 Notes

This repository focuses on method implementation, while datasets and trained models are hosted separately on Hugging Face.

The code is intended for research and academic use.

For reproducibility, users are strongly encouraged to follow the provided environment configuration.

📄 License

This project is released under the MIT License.
See the LICENSE file for details.


📬 Contact

For questions, issues, or collaborations, please contact:

Bo Wang
Email: wangbo@ihb.ac.cn

