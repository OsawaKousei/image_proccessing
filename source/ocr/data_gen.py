import random

import cv2
import numpy as np

BACKGROUND = "/home/kousei/image_proccessing/source/ocr/free-texture.jpg"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2
FONT_THICKNESS = 2
FONT_COLOR = (255, 255, 255)
FONT_LINE_TYPE = cv2.LINE_AA
FONT_MARGIN = 10
TEXT_X = 50
TEXT_Y = 150
IMG_PATH = "/home/kousei/image_proccessing/source/ocr/dataset/"


# アルミの刻印画像を生成するクラス
class Stamped:
    def __init__(self) -> None:
        pass

    def __draw_text(self, text: str, name: str) -> None:
        # 背景画像の読み込み
        self.img = cv2.imread(BACKGROUND)
        # 背景画像をやや暗くする
        self.img = cv2.addWeighted(
            self.img, 0.5, np.zeros_like(self.img), 0.5, 0
        )
        # テキストの挿入
        cv2.putText(
            self.img,
            text,
            (TEXT_X, TEXT_Y),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            FONT_LINE_TYPE,
        )
        # 解像度を下げる(アスペクト比約3:2)
        self.img = cv2.resize(self.img, (192, 129))
        # 画像の保存
        cv2.imwrite(name, self.img)

    def __generate_text(self) -> str:
        # ランダムな２桁の数字を生成
        text = str(random.randint(10, 99))
        # ランダムな一つのアルファベットを生成
        text += chr(random.randint(65, 90))
        # スペースを追加
        text += " "
        # ランダムなアルファベットを生成
        text += chr(random.randint(65, 90))
        text += chr(random.randint(65, 90))
        return text

    def generate(self, num: int) -> None:
        for i in range(num):
            text = self.__generate_text()
            name = IMG_PATH + "data" + str(i) + ".jpg"
            self.__draw_text(text, name)


if __name__ == "__main__":
    # 初期化
    stamped = Stamped()
    # 画像生成
    stamped.generate(10)
