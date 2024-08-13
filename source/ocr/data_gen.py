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

TRAIN_PATH = "/home/kousei/image_proccessing/source/ocr/train"
VAILD_PATH = "/home/kousei/image_proccessing/source/ocr/vaild"


# アルミの刻印画像を生成するクラス
class Stamped:
    def __init__(self) -> None:
        pass

    def __draw_text(self, text: str, name: str) -> None:
        # 背景画像の読み込み
        self.img = cv2.imread(BACKGROUND)
        # 背景画像をやや暗くする
        self.img = cv2.addWeighted(
            self.img, 0.7, np.zeros_like(self.img), 0.7, 0
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
        # テキストを囲うバウンディングボックスを取得
        text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_w, text_h = text_size
        text_x = TEXT_X - FONT_MARGIN
        text_y = TEXT_Y - text_h - FONT_MARGIN
        text_w += 2 * FONT_MARGIN
        text_h += 2 * FONT_MARGIN
        cv2.rectangle(
            self.img,
            (text_x, text_y),
            (text_x + text_w, text_y + text_h),
            (0, 255, 0),
            2,
        )

        # テキストとバウンディングボックスの座標を保存
        self.text = text
        self.text_x = text_x
        self.text_y = text_y
        self.text_w = text_w
        self.text_h = text_h

        # テキストとバウンディングボックスをランダムに回転
        angle = random.randint(-30, 30)
        center = (TEXT_X + text_w // 2, TEXT_Y - text_h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.img = cv2.warpAffine(
            self.img, rot_mat, (self.img.shape[1], self.img.shape[0])
        )

        # 回転後のバウンディングボックスを算出
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        corners = np.array(
            [
                [text_x, text_y],
                [text_x + text_w, text_y],
                [text_x + text_w, text_y + text_h],
                [text_x, text_y + text_h],
            ],
            dtype=np.float32,
        )
        corners = np.dot(rot_mat, np.hstack([corners, np.ones((4, 1))]).T).T
        corners = corners.reshape(-1, 2).astype(np.int32)
        x, y, w, h = cv2.boundingRect(corners)
        self.text_x = x
        self.text_y = y
        self.text_w = w
        self.text_h = h

        # 解像度を下げる(アスペクト比約3:2)
        self.img = cv2.resize(self.img, (192, 129))
        # ノイズを追加
        noise = np.random.normal(0, 3, (129, 192, 3))
        self.img = np.clip(self.img + noise, 0, 255).astype(np.uint8)
        # テキストを囲う矩形領域を算出
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

    def generate_img(self, num: int) -> None:
        for i in range(num):
            text = self.__generate_text()
            name = IMG_PATH + "data" + str(i) + ".jpg"
            self.__draw_text(text, name)

    def generate_yolo_dataset(self, num: int) -> None:
        train_num = int(num * 0.8)
        valid_num = num - train_num

        for i in range(train_num):
            text = self.__generate_text()
            name = TRAIN_PATH + "/images/" + str(i) + ".jpg"
            self.__draw_text(text, name)
            # yoloのラベルファイルを生成
            with open(TRAIN_PATH + "/labels/" + str(i) + ".txt", "w") as f:
                f.write(
                    "0 "
                    + str(self.text_x)
                    + " "
                    + str(self.text_y)
                    + " "
                    + str(self.text_w)
                    + " "
                    + str(self.text_h)
                )

        for i in range(valid_num):
            text = self.__generate_text()
            name = VAILD_PATH + "/images/" + str(i) + ".jpg"
            self.__draw_text(text, name)
            # yoloのラベルファイルを生成
            with open(VAILD_PATH + "/labels/" + str(i) + ".txt", "w") as f:
                f.write(
                    "0 "
                    + str(self.text_x)
                    + " "
                    + str(self.text_y)
                    + " "
                    + str(self.text_w)
                    + " "
                    + str(self.text_h)
                )


if __name__ == "__main__":
    # 初期化
    stamped = Stamped()
    # 画像生成
    stamped.generate_yolo_dataset(1000)
