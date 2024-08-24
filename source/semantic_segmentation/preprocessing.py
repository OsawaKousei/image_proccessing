# パッケージのimport
import os.path as osp

import torch.utils.data as data
from PIL import Image

# データ処理のクラスとデータオーギュメンテーションのクラスをimportする
from utils.data_augumentation import (
    Compose,
    Normalize_Tensor,
    RandomMirror,
    RandomRotation,
    Resize,
    Scale,
)


def make_datapath_list(rootpath):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_template = osp.join(rootpath, "SegmentationClass", "%s.png")

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + "ImageSets/Segmentation/train.txt")
    val_id_names = osp.join(rootpath + "ImageSets/Segmentation/val.txt")

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = imgpath_template % file_id  # 画像のパス
        anno_path = annopath_template % file_id  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = imgpath_template % file_id  # 画像のパス
        anno_path = annopath_template % file_id  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# 動作確認 ファイルパスのリストを取得
rootpath = "./data/VOCdevkit/VOC2012/"

train_img_list, train_anno_list, val_img_list, val_anno_list = (
    make_datapath_list(rootpath=rootpath)
)

print(train_img_list[0])
print(train_anno_list[0])


class DataTransform:
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose(
                [
                    Scale(scale=[0.5, 1.5]),  # 画像の拡大
                    RandomRotation(angle=[-10, 10]),  # 回転
                    RandomMirror(),  # ランダムミラー
                    Resize(input_size),  # リサイズ(input_size)
                    Normalize_Tensor(
                        color_mean, color_std
                    ),  # 色情報の標準化とテンソル化
                ]
            ),
            "val": Compose(
                [
                    Resize(input_size),  # リサイズ(input_size)
                    Normalize_Tensor(
                        color_mean, color_std
                    ),  # 色情報の標準化とテンソル化
                ]
            ),
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        """画像の枚数を返す"""
        return len(self.img_list)

    def __getitem__(self, index):
        """
        前処理をした画像のTensor形式のデータとアノテーションを取得
        """
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        """画像のTensor形式のデータ、アノテーションを取得する"""

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)  # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)  # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


# 動作確認

# (RGB)の色の平均値と標準偏差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

# データセット作成
train_dataset = VOCDataset(
    train_img_list,
    train_anno_list,
    phase="train",
    transform=DataTransform(
        input_size=475, color_mean=color_mean, color_std=color_std
    ),
)

val_dataset = VOCDataset(
    val_img_list,
    val_anno_list,
    phase="val",
    transform=DataTransform(
        input_size=475, color_mean=color_mean, color_std=color_std
    ),
)

# データの取り出し例
print(val_dataset.__getitem__(0)[0].shape)
print(val_dataset.__getitem__(0)[1].shape)
print(val_dataset.__getitem__(0))

# データローダーの作成

batch_size = 8

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
imges, anno_class_imges = next(batch_iterator)  # 1番目の要素を取り出す
print(imges.size())  # torch.Size([8, 3, 475, 475])
print(anno_class_imges.size())  # torch.Size([8, 3, 475, 475])
