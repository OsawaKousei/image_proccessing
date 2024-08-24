import cv2
from ultralytics import YOLO

model = YOLO(
    "/home/kousei/image_proccessing/runs/detect/train15/weights/best.pt"
)  # best.ptまでの相対パス

results = model('/home/kousei/image_proccessing/confidencial/IMG_1021 2.jpg', save=True)
