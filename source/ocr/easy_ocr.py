import easyocr
from PIL import Image, ImageDraw

reader = easyocr.Reader(["ch_sim", "en"])


def analyze_picture(target_path: str) -> None:
    draw_chararea(target_path, reader.readtext(target_path))


# <追加>入力画像内に文字列の領域を赤枠で囲う
def draw_chararea(target_path: str, results) -> None:
    image = Image.open(target_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # 座標情報からテキスト領域を四角で囲う
    for result in results:
        print(result)
        p0, p1, p2, p3 = result[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill="red", width=3)
    image.save("draw_chararea.png")


if __name__ == "__main__":
    target_path = "/home/kousei/image_proccessing/confidencial/text.jpg"
    analyze_picture(target_path)
