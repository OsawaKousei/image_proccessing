import cv2
from ultralytics import YOLO

model = YOLO(
    "/home/kousei/image_proccessing/source/yolo/runs/detect/train3/weights/best.pt"
)  # best.ptまでの相対パス
video_path = "./path_to_video/video_name.mp4"  # テストしたい動画
cap = cv2.VideoCapture(video_path)
annotated_frames = []
while cap.isOpened():  # フレームがある間繰り返す
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True)  # 物体をトラッキング
        annotated_frame = results[0].plot()
    else:
        break
    annotated_frames.append(annotated_frame)
# annotated_framesをmp4として保存
height, width, layers = annotated_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("./annotated_video.mp4", fourcc, 30, (width, height))
for frame in annotated_frames:
    video.write(frame)
video.release()
cap.release()
cv2.destroyAllWindows()
print("finished")
