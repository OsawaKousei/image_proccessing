import cv2

# 画像の読み込み
image = cv2.imread("/home/kousei/image_proccessing/confidencial/box2.jpg")
# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# メディアンフィルタでノイズ除去
denoised = cv2.medianBlur(gray, 3)
# denoised = gray

# # 適応的二値化処理
# _, binary = cv2.threshold(
#     denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )
binary = denoised

# 保存
cv2.imwrite(
    "/home/kousei/image_proccessing/confidencial/binary_box2.jpg", binary
)
