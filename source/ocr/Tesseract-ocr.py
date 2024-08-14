import pytesseract
from PIL import Image

# 画像ファイルを開く
image = Image.open("/home/kousei/image_proccessing/confidencial/binary.jpg")

# Tesseract-OCRでテキストを抽出
text = pytesseract.image_to_string(image, lang="eng")

print(text)
