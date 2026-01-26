from ultralytics import YOLO

model = YOLO("./runs6/detect/SHE/weights/best.pt")

results = model.track(
    source="/home/bwang/tryy.mp4",
    tracker="cbt.yaml",
    iou=0.45,
    save=True)
  