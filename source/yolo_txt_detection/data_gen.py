import random

import cv2
import numpy as np
import pandas as pd

BACKGROUND = "/home/kousei/image_proccessing/source/ocr/free-texture.jpg"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
FONT_COLOR = (255, 255, 255)
FONT_LINE_TYPE = cv2.LINE_AA
TEXT_X = 25
TEXT_Y = 75
IMG_PATH = "/home/kousei/image_proccessing/source/ocr/dataset/"

IMG_H = 129
IMG_W = 192

TRAIN_PATH = "/home/kousei/image_proccessing/source/ocr/train"
VAILD_PATH = "/home/kousei/image_proccessing/source/ocr/valid"


# アルミの刻印画像を生成するクラス
class Stamped:
    def __init__(self) -> None:
        self.label = pd.DataFrame(
            columns=["label", "text", "x", "y", "w", "h"]
        )

    def __draw_text(self, text: str, name: str) -> None:
        # 背景画像の読み込み
        self.img = cv2.imread(BACKGROUND)
        # 背景画像をやや暗くする
        self.img = cv2.addWeighted(
            self.img, 0.7, np.zeros_like(self.img), 0.7, 0
        )
        # 解像度を下げる(アスペクト比約3:2)
        self.img = cv2.resize(self.img, (IMG_W, IMG_H))
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
        text_x = TEXT_X + text_w // 2
        text_y = TEXT_Y - text_h // 2

        # テキストとバウンディングボックスの座標を保存
        self.text = text

        # # テキストとバウンディングボックスをランダムに回転
        # angle = random.randint(-30, 30)
        # center = (TEXT_X + text_w // 2, TEXT_Y - text_h // 2)
        # rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # self.img = cv2.warpAffine(
        #     self.img, rot_mat, (self.img.shape[1], self.img.shape[0])
        # )

        # # 回転後のバウンディングボックスを算出
        # rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # corners = np.array(
        #     [
        #         [text_x, text_y],
        #         [text_x + text_w, text_y],
        #         [text_x + text_w, text_y + text_h],
        #         [text_x, text_y + text_h],
        #     ],
        #     dtype=np.float32,
        # )
        # corners = np.dot(rot_mat, np.hstack([corners, np.ones((4, 1))]).T).T
        # corners = corners.reshape(-1, 2).astype(np.int32)
        # x, y, w, h = cv2.boundingRect(corners)

        # バウンディングボックスの座標を保存
        x, y, w, h = text_x, text_y, text_w, text_h

        # バウンディングボックスを描画
        # cv2.rectangle(
        #     self.img,
        #     (text_x, text_y),
        #     (text_x + text_w, text_y + text_h),
        #     (0, 255, 0),
        #     2,
        # )

        # print("raw",x, y, w, h)

        # バウンディングボックスの座標を正規化
        x = x / IMG_W
        y = y / IMG_H
        w = w / IMG_W
        h = h / IMG_H

        # print("normalized",x, y, w, h)

        # print(x, y, w, h)

        # labelに追加
        frame = pd.DataFrame(
            [[0, text, x, y, w, h]],
            columns=["label", "text", "x", "y", "w", "h"],
        )
        self.label = pd.concat([self.label, frame])

        # ノイズを追加
        noise = np.random.normal(0, 3, (IMG_H, IMG_W, 3))
        self.img = np.clip(self.img + noise, 0, 255).astype(np.uint8)
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

        # labelのインデックスを振り直す
        self.label = self.label.reset_index(drop=True)

        print(self.label.head())
        # yoloのラベルファイルを生成
        for i in range(train_num):
            with open(TRAIN_PATH + "/labels/" + str(i) + ".txt", "w") as f:
                f.write(
                    "0 "
                    + str(self.label["x"][i])
                    + " "
                    + str(self.label["y"][i])
                    + " "
                    + str(self.label["w"][i])
                    + " "
                    + str(self.label["h"][i])
                )

        # labelファイルを初期化
        self.label = pd.DataFrame(columns=["text", "x", "y", "w", "h"])

        for i in range(valid_num):
            text = self.__generate_text()
            name = VAILD_PATH + "/images/" + str(i) + ".jpg"
            self.__draw_text(text, name)

        # labelのインデックスを振り直す
        self.label = self.label.reset_index(drop=True)

        # yoloのラベルファイルを生成
        for i in range(valid_num):
            with open(VAILD_PATH + "/labels/" + str(i) + ".txt", "w") as f:
                f.write(
                    "0 "
                    + str(self.label["x"][i])
                    + " "
                    + str(self.label["y"][i])
                    + " "
                    + str(self.label["w"][i])
                    + " "
                    + str(self.label["h"][i])
                )


if __name__ == "__main__":
    # 初期化
    stamped = Stamped()
    # 画像生成
    stamped.generate_yolo_dataset(1000)
