from paddleocr import PaddleOCR
from PIL import Image, ImageColor, ImageDraw

# PaddleOCRのインスタンス化（日本語対応）
ocr = PaddleOCR(lang="en")

# 画像ファイルからテキストを抽出
results = ocr.ocr(
    "/home/kousei/image_proccessing/confidencial/binary_box2.jpg"
)

# 抽出したテキストを表示
for line in results:
    print(line)

target_path = "/home/kousei/image_proccessing/confidencial/binary_box2.jpg"
image = Image.open(target_path).convert("RGB")
draw = ImageDraw.Draw(image)
# バウンディングボックスを描画
for result in results:
    print(result)
    print(result[0])
    print(result[0][0])
    p0 = result[0][0][0]
    p1 = result[0][0][1]
    p2 = result[0][0][2]
    p3 = result[0][0][3]
    draw.line([*p0, *p1, *p2, *p3, *p0], fill="red", width=3)
image.save("draw_chararea.png")
