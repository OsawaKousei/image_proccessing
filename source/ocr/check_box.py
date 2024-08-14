import cv2
IMG_H = 129
IMG_W = 192

img = cv2.imread('/home/kousei/image_proccessing/source/ocr/valid/images/0.jpg')
# 解像度を下げる(アスペクト比約3:2)
img = cv2.resize(img, (IMG_W, IMG_H))
# labelの読み込み
with open('/home/kousei/image_proccessing/source/ocr/valid/labels/0.txt', 'r') as f:
    labels = f.readlines()
# x,y は中心座標、w,hは幅と高さ
for label in labels:
    label = label.split()
    x = int(float(label[1]) * img.shape[1])
    y = int(float(label[2]) * img.shape[0])
    w = int(float(label[3]) * img.shape[1])
    h = int(float(label[4]) * img.shape[0])
    print(x, y, w, h)
    cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

# 保存
cv2.imwrite('/home/kousei/image_proccessing/source/ocr/valid/images/0_box.jpg', img)
