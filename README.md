```markdown
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
