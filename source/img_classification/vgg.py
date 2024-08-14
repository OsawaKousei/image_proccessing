# パッケージのimport
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import models, transforms

# PyTorchのバージョン確認
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# 学習済みのVGG-16モデルをロード
# 初めて実行する際は、学習済みパラメータをダウンロードするため、実行に時間がかかります

# VGG-16モデルのインスタンスを生成
use_pretrained = True  # 学習済みのパラメータを使用
net = models.vgg16(pretrained=use_pretrained)
net.eval()  # 推論モードに設定

# モデルのネットワーク構成を出力
print(net)


# 入力画像の前処理のクラス
class BaseTransform:
    """
    画像のサイズをリサイズし、色を標準化する。

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(
                    resize
                ),  # 短い辺の長さがresizeの大きさになる
                transforms.CenterCrop(
                    resize
                ),  # 画像中央をresize × resizeで切り取り
                transforms.ToTensor(),  # Torchテンソルに変換
                transforms.Normalize(mean, std),  # 色情報の標準化
            ]
        )

    def __call__(self, img):
        return self.base_transform(img)


# 画像前処理の動作を確認

# 1. 画像読み込み
image_file_path = "/home/kousei/image_proccessing/source/img_classification/data/goldenretriever-3724972_640.jpg"
img = Image.open(image_file_path)  # [高さ][幅][色RGB]

# 2. 元の画像の表示
plt.imshow(img)
plt.show()

# 3. 画像の前処理と処理済み画像の表示
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)  # torch.Size([3, 224, 224])

# (色、高さ、幅)を (高さ、幅、色)に変換し、0-1に値を制限して表示
img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.savefig(
    "/home/kousei/image_proccessing/source/img_classification/data/goldenretriever-3724972_640_transformed.jpg"
)


# 出力結果からラベルを予測する後処理クラス
class ILSVRCPredictor:
    """
    ILSVRCデータに対するモデルの出力からラベルを求める。

    Attributes
    ----------
    class_index : dictionary
            クラスindexとラベル名を対応させた辞書型変数。
    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        """
        確率最大のILSVRCのラベル名を取得する。

        Parameters
        ----------
        out : torch.Size([1, 1000])
            Netからの出力。

        Returns
        -------
        predicted_label_name : str
            最も予測確率が高いラベルの名前
        """
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name


# ILSVRCのラベル情報をロードし辞意書型変数を生成します
ILSVRC_class_index = json.load(
    open(
        "/home/kousei/image_proccessing/source/img_classification/data/imagenet_class_index.json",
        "r",
    )
)


# ILSVRCPredictorのインスタンスを生成します
predictor = ILSVRCPredictor(ILSVRC_class_index)

# 入力画像を読み込む
image_file_path = "/home/kousei/image_proccessing/source/img_classification/data/goldenretriever-3724972_640.jpg"
img = Image.open(image_file_path)  # [高さ][幅][色RGB]

# 前処理の後、バッチサイズの次元を追加する
transform = BaseTransform(resize, mean, std)  # 前処理クラス作成
img_transformed = transform(img)  # torch.Size([3, 224, 224])
inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])

# モデルに入力し、モデル出力をラベルに変換する
out = net(inputs)  # torch.Size([1, 1000])
result = predictor.predict_max(out)

# 予測結果を出力する
print("入力画像の予測結果：", result)
