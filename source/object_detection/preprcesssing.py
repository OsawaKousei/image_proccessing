# パッケージのimport
import os.path as osp
import random

# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

# フォルダ「utils」にあるdata_augumentation.pyからimport。
# 入力画像の前処理をするクラス
from utils.data_augumentation import (
    Compose,
    ConvertFromInts,
    Expand,
    PhotometricDistort,
    RandomMirror,
    RandomSampleCrop,
    Resize,
    SubtractMeans,
    ToAbsoluteCoords,
    ToPercentCoords,
)

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する


def make_datapath_list(
    rootpath: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    データへのパスを格納したリストを作成する。

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
    annopath_template = osp.join(rootpath, "Annotations", "%s.xml")

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + "ImageSets/Main/train.txt")
    val_id_names = osp.join(rootpath + "ImageSets/Main/val.txt")

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = imgpath_template % file_id  # 画像のパス
        anno_path = annopath_template % file_id  # アノテーションのパス
        train_img_list.append(img_path)  # リストに追加
        train_anno_list.append(anno_path)  # リストに追加

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = imgpath_template % file_id  # 画像のパス
        anno_path = annopath_template % file_id  # アノテーションのパス
        val_img_list.append(img_path)  # リストに追加
        val_anno_list.append(anno_path)  # リストに追加

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# ファイルパスのリストを作成
rootpath = "/home/kousei/image_proccessing/source/object_detection/data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = (
    make_datapath_list(rootpath)
)

# 動作確認
print(train_img_list[0])

# 「XML形式のアノテーション」を、リスト形式に変換するクラス


class Anno_xml2list(object):
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

    Attributes
    ----------
    classes : リスト
        VOCのクラス名を格納したリスト
    """

    def __init__(self, classes: list[str]):
        self.classes = classes

    def __call__(
        self, xml_path: str, width: int, height: int
    ) -> list[list[float]]:
        """
        Convert XML annotation data for a single image to a list format, normalized by image size.

        Parameters
        ----------
        xml_path : str
            Path to the XML file.
        width : int
            Width of the target image.
        height : int
            Height of the target image.

        Returns
        -------
        ret : list[list[float]]
            List containing the annotation data for each object in the image. Each element represents a single object and contains [xmin, ymin, xmax, ymax, label_ind].
        """

        ret = []

        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)  # type: ignore
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find("name").text.lower().strip()  # type: ignore
            bbox = obj.find("bndbox")  # type: ignore

            pts = ["xmin", "ymin", "xmax", "ymax"]

            for pt in pts:
                cur_pixel = float(int(bbox.find(pt).text) - 1)  # type: ignore

                if pt == "xmin" or pt == "xmax":
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            ret += [bndbox]

        return ret


# 動作確認
voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

transform_anno = Anno_xml2list(voc_classes)

# 画像の読み込み OpenCVを使用
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
height, width, channels = img.shape  # 画像のサイズを取得

# アノテーションをリストで表示
print(transform_anno(val_anno_list[ind], width, height))


class DataTransform:
    """
    画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
    画像のサイズを300x300にする。
    学習時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (B, G, R)
        各色チャネルの平均値。
    """

    def __init__(self, input_size: int, color_mean: tuple[int, int, int]):
        self.data_transform = {
            "train": Compose(
                [
                    ConvertFromInts(),  # intをfloat32に変換
                    ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
                    PhotometricDistort(),  # 画像の色調などをランダムに変化
                    Expand(color_mean),  # 画像のキャンバスを広げる
                    RandomSampleCrop(),  # 画像内の部分をランダムに抜き出す
                    RandomMirror(),  # 画像を反転させる
                    ToPercentCoords(),  # アノテーションデータを0-1に規格化
                    Resize(
                        input_size
                    ),  # 画像サイズをinput_size×input_sizeに変形
                    SubtractMeans(color_mean),  # BGRの色の平均値を引き算
                ]
            ),
            "val": Compose(
                [
                    ConvertFromInts(),  # intをfloatに変換
                    Resize(
                        input_size
                    ),  # 画像サイズをinput_size×input_sizeに変形
                    SubtractMeans(color_mean),  # BGRの色の平均値を引き算
                ]
            ),
        }

    def __call__(
        self,
        img: np.ndarray,
        phase: str,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, boxes, labels)  # type: ignore


# 動作の確認

# 1. 画像読み込み
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
height, width, channels = img.shape  # 画像のサイズを取得

# 2. アノテーションをリストに
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)

# 3. 元画像の表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig("original_image.png")

# 4. 前処理クラスの作成
color_mean = (104, 117, 123)  # (BGR)の色の平均値
input_size = 300  # 画像のinputサイズを300×300にする
transform = DataTransform(input_size, color_mean)

print(anno_list)

anno_array = np.array(anno_list)

# 5. train画像の表示
phase = "train"
img_transformed, boxes, labels = transform(
    img, phase, anno_array[:, :4], anno_array[:, 4]
)
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.savefig("train_image.png")


# 6. val画像の表示
phase = "val"
img_transformed, boxes, labels = transform(
    img, phase, anno_array[:, :4], anno_array[:, 4]
)
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.savefig("val_image.png")

# VOC2012のDatasetを作成する


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
    transform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """

    def __init__(
        self,
        img_list: list[str],
        anno_list: list[str],
        phase: str,
        transform: object,
        transform_anno: object,
    ):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # train もしくは valを指定
        self.transform = transform  # 画像の変形
        self.transform_anno = (
            transform_anno  # アノテーションデータをxmlからリストへ
        )

    def __len__(self) -> int:
        """画像の枚数を返す"""
        return len(self.img_list)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        前処理をした画像のテンソル形式のデータとアノテーションを取得
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(
        self, index: int
    ) -> tuple[torch.Tensor, np.ndarray, int, int]:
        """前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する"""

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得

        # 2. xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = np.array(
            self.transform_anno(anno_file_path, width, height)  # type: ignore
        )

        # 3. 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]  # type: ignore
        )

        # 色チャネルの順番がBGRになっているので、RGBに順番変更
        # さらに（高さ、幅、色チャネル）の順を（色チャネル、高さ、幅）に変換
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # BBoxとラベルをセットにしたnp.arrayを作成、変数名「gt」はground truth（答え）の略称
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


# 動作確認
color_mean = (104, 117, 123)  # (BGR)の色の平均値
input_size = 300  # 画像のinputサイズを300×300にする

train_dataset = VOCDataset(
    train_img_list,
    train_anno_list,
    phase="train",
    transform=DataTransform(input_size, color_mean),
    transform_anno=Anno_xml2list(voc_classes),
)

val_dataset = VOCDataset(
    val_img_list,
    val_anno_list,
    phase="val",
    transform=DataTransform(input_size, color_mean),
    transform_anno=Anno_xml2list(voc_classes),
)


# データの取り出し例
print(val_dataset.__getitem__(1))


def od_collate_fn(
    batch: list[tuple[torch.Tensor, np.ndarray]]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
    この変化に対応したDataLoaderを作成するために、
    カスタイマイズした、collate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像imgです
        targets.append(
            torch.FloatTensor(sample[1])
        )  # sample[1] はアノテーションgtです

    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリストです。
    # リストのサイズはミニバッチサイズです。
    # リストtargetsの要素は [n, 5] となっています。
    # nは画像ごとに異なり、画像内にある物体の数となります。
    # 5は [xmin, ymin, xmax, ymax, class_index] です

    return imgs, targets


# データローダーの作成

batch_size = 4

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=od_collate_fn,
)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn
)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す
print(images.size())  # torch.Size([4, 3, 300, 300])
print(len(targets))
print(
    targets[1].size()
)  # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数

print(train_dataset.__len__())
print(val_dataset.__len__())
