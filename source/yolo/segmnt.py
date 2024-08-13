from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8x-seg.pt")

    # Predict the model
    model.predict('https://ultralytics.com/images/bus.jpg', save=True, conf=0.5, boxes=False)
