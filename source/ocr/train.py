from ultralytics import YOLO

model = YOLO(
    "yolov8n.pt"
)  # yolov8n/s/m/l/xのいずれかを指定。多クラスの検出であるほど大きいパラメータが必要
model.train(
    data="/home/kousei/image_proccessing/source/ocr/data.yaml",
    epochs=20,
    batch=20,
)  # 先ほど作成したデータセット内のyamlファイルまでのパスを指定
