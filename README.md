# STAFDD: A Spatio-Temporal Automatic Fish Disease Detection Method

This repository provides the **official implementation** of the paper:

**STAFDD: A Spatio-Temporal Automatic Fish Disease Detection Method**  
Bo Wang et al.

This project focuses on automated fish disease detection in high-density aquaculture environments by integrating **object detection, multi-target tracking, and spatio-temporal behavior analysis**.

---

## 🔍 Overview

STAFDD is a spatio-temporal automatic fish disease detection framework designed for complex aquaculture scenarios.  
The framework integrates:

- An improved YOLOv8-based detector (**SSCA-YOLO**)
- A customized multi-object tracking algorithm (**ByTeSort**)
- Trajectory-level behavior analysis using **LSTM**
- Re-identification (ReID) for long-term identity consistency

---

## ⚙️ Environment Setup

The experiments were conducted under the following environment:

- **Operating System**: Linux 5.14.0-570.32.1.el9_6 (x86_64)
- **Python**: 3.12.9
- **PyTorch**: 2.7.0 + CUDA 12.6
- **CUDA**: 12.6 (available)
- **cuDNN**: 90501

### Key Dependencies

```text
ultralytics           8.3.127
ultralytics-thop      2.0.14
⚠️ Note
The codebase is developed and tested under the above configuration.
Compatibility with other versions is not guaranteed.

📁 Repository Structure
The repository is organized as follows:

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

---

🔧 Core Components
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

---

📊 Dataset
The YOLO training dataset, pretrained model weights, test videos, and test results are publicly available at:

👉 Hugging Face Dataset
https://huggingface.co/datasets/wangbo66/STAFDD-dataset

Please refer to the dataset card on Hugging Face for detailed data organization and usage instructions.

🚀 Usage
Training
Training is performed using the SSCA-YOLO framework defined in SSCA-YOLO.yaml, with parameters specified in tran_yolo.py.

python tran_yolo.py
Tracking and Behavior Analysis
Detection results are processed using the proposed ByTeSort tracker.

Fish trajectories are further analyzed using the LSTM-based scripts in the LSTM/ directory to infer fish health status.

📄 License
This project is released under the MIT License.
See the LICENSE file for details.

📖 Citation
If you use this code or dataset in your research, please cite the corresponding paper:

@article{wang2024stafdd,
  title={STAFDD: A Spatio-Temporal Automatic Fish Disease Detection Method},
  author={Wang, Bo and others},
  journal={},
  year={2024}
}
📬 Contact
For questions or collaborations, please contact:

Bo Wang
Email: 3020201781@jsnu.edu
