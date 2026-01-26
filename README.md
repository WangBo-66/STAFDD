<div align="center">
  <p>
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="STAFDD banner">
  </p>

[English](README.md)

<div>
    <img src="https://img.shields.io/badge/Task-Fish%20Disease%20Detection-blue" />
    <img src="https://img.shields.io/badge/Model-SSCA--YOLO-green" />
    <img src="https://img.shields.io/badge/Tracking-ByTeSort-orange" />
    <img src="https://img.shields.io/badge/License-CC--BY--4.0-lightgrey" />
</div>
</div>

<br>

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
