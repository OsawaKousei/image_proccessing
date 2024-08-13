import random

import cv2
import numpy as np


# アルミの刻印画像を生成するクラス
class Stamped:
    def __init__(self) -> None:
        self.img = None
        self.text = None

    def draw_text(self, text: str) -> None:
        # implementation of draw_text method
        pass

    def generate_text(self) -> str:
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
            self.text = self.__generate_text()
            self.__draw_text(self.text)
            self.img.save(f"stamped_{i}.png")


if __name__ == "__main__":
    # 初期化
    stamped = Stamped()
    print(stamped.generate_text())
