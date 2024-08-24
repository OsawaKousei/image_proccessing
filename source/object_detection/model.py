# パッケージのimport
from itertools import product
from math import sqrt
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function


# 35層にわたる、vggモジュールを作成
def make_vgg() -> nn.ModuleList:
    layers = []
    in_channels = 3  # 色チャネル数

    # vggモジュールで使用する畳み込み層やマックスプーリングのチャネル数
    cfg: list[int | str] = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "MC",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ]

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "MC":
            # ceilは出力サイズを、計算結果（float）に対して、切り上げで整数にするモード
            # デフォルトでは出力サイズを計算結果（float）に対して、切り下げで整数にするfloorモード
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5,
        conv6,
        nn.ReLU(inplace=True),
        conv7,
        nn.ReLU(inplace=True),
    ]
    return nn.ModuleList(layers)


# 動作確認
vgg_test = make_vgg()
print(vgg_test)


# 8層にわたる、extrasモジュールを作成
def make_extras() -> nn.ModuleList:
    layers = []
    in_channels = (
        1024  # vggモジュールから出力された、extraに入力される画像チャネル数
    )

    # extraモジュールの畳み込み層のチャネル数を設定するコンフィギュレーション
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    # 活性化関数のReLUは今回はSSDモデルの順伝搬のなかで用意することにし、
    # extraモジュールでは用意していません

    return nn.ModuleList(layers)


# 動作確認
extras_test = make_extras()
print(extras_test)

# デフォルトボックスのオフセットを出力するloc_layers、
# デフォルトボックスに対する各クラスの信頼度confidenceを出力するconf_layersを作成


def make_loc_conf(
    num_classes: int = 21, bbox_aspect_num: list[int] = [4, 6, 6, 6, 4, 4]
) -> tuple[nn.ModuleList, nn.ModuleList]:

    loc_layers = []
    conf_layers = []

    # VGGの22層目、conv4_3（source1）に対する畳み込み層
    loc_layers += [
        nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)
    ]
    conf_layers += [
        nn.Conv2d(
            512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1
        )
    ]

    # VGGの最終層（source2）に対する畳み込み層
    loc_layers += [
        nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)
    ]
    conf_layers += [
        nn.Conv2d(
            1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1
        )
    ]

    # extraの（source3）に対する畳み込み層
    loc_layers += [
        nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)
    ]
    conf_layers += [
        nn.Conv2d(
            512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1
        )
    ]

    # extraの（source4）に対する畳み込み層
    loc_layers += [
        nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)
    ]
    conf_layers += [
        nn.Conv2d(
            256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1
        )
    ]

    # extraの（source5）に対する畳み込み層
    loc_layers += [
        nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)
    ]
    conf_layers += [
        nn.Conv2d(
            256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1
        )
    ]

    # extraの（source6）に対する畳み込み層
    loc_layers += [
        nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)
    ]
    conf_layers += [
        nn.Conv2d(
            256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1
        )
    ]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


# 動作確認
loc_test, conf_test = make_loc_conf()
print(loc_test)
print(conf_test)


# convC4_3からの出力をscale=20のL2Normで正規化する層
class L2Norm(nn.Module):
    def __init__(self, input_channels: int = 512, scale: int = 20):
        super(L2Norm, self).__init__()  # 親クラスのコンストラクタ実行
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # 係数weightの初期値として設定する値
        self.reset_parameters()  # パラメータの初期化
        self.eps = 1e-10

    def reset_parameters(self) -> None:
        """結合パラメータを大きさscaleの値にする初期化を実行"""
        init.constant_(
            self.weight, self.scale
        )  # weightの値がすべてscale（=20）になる

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """38×38の特徴量に対して、512チャネルにわたって2乗和のルートを求めた
        38×38個の値を使用し、各特徴量を正規化してから係数をかけ算する層"""

        # 各チャネルにおける38×38個の特徴量のチャネル方向の2乗和を計算し、
        # さらにルートを求め、割り算して正規化する
        # normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])になります
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # 係数をかける。係数はチャネルごとに1つで、512個の係数を持つ
        # self.weightのテンソルサイズはtorch.Size([512])なので
        # torch.Size([batch_num, 512, 38, 38])まで変形します
        weights = (
            self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        )
        out = weights * x

        return out


# デフォルトボックスを出力するクラス
class DBox(object):
    def __init__(self, cfg: dict[str, Any]):
        super(DBox, self).__init__()

        # 初期設定
        self.image_size: int = cfg["input_size"]  # 画像サイズの300
        # [38, 19, …] 各sourceの特徴量マップのサイズ
        self.feature_maps: list[int] = cfg["feature_maps"]
        self.num_priors: int = len(cfg["feature_maps"])  # sourceの個数=6
        self.steps: list[int] = cfg["steps"]  # [8, 16, …] DBoxのピクセルサイズ

        self.min_sizes: list[int] = cfg["min_sizes"]
        # [30, 60, …] 小さい正方形のDBoxのピクセルサイズ（正確には面積）

        self.max_sizes: list[int] = cfg["max_sizes"]
        # [60, 111, …] 大きい正方形のDBoxのピクセルサイズ（正確には面積）

        self.aspect_ratios: list[list[int]] = cfg[
            "aspect_ratios"
        ]  # 長方形のDBoxのアスペクト比

    def make_dbox_list(self) -> torch.Tensor:
        """DBoxを作成する"""
        mean: list[float] = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(
                range(f), repeat=2
            ):  # fまでの数で2ペアの組み合わせを作る　f_P_2 個
                # 特徴量の画像サイズ
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k: int = int(self.image_size / self.steps[k])

                # DBoxの中心座標 x,y　ただし、0～1で規格化している
                cx: float = (j + 0.5) / f_k
                cy: float = (i + 0.5) / f_k

                # アスペクト比1の小さいDBox [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k: float = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # アスペクト比1の大きいDBox [cx,cy, width, height]
                # 'max_sizes': [60, 111, 162, 213, 264, 315],
                s_k_prime: float = sqrt(
                    s_k * (self.max_sizes[k] / self.image_size)
                )
                mean += [cx, cy, s_k_prime, s_k_prime]

                # その他のアスペクト比のdefBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # DBoxをテンソルに変換 torch.Size([8732, 4])
        output: torch.Tensor = torch.Tensor(mean).view(-1, 4)

        # DBoxが画像の外にはみ出るのを防ぐため、大きさを最小0、最大1にする
        output.clamp_(max=1, min=0)

        return output


# 動作の確認

# SSD300の設定
ssd_cfg = {
    "num_classes": 21,  # 背景クラスを含めた合計クラス数
    "input_size": 300,  # 画像の入力サイズ
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# DBox作成
dbox = DBox(ssd_cfg)
dbox_list = dbox.make_dbox_list()

# DBoxの出力を確認する
print(pd.DataFrame(dbox_list.numpy()))


# オフセット情報を使い、DBoxをBBoxに変換する関数


def decode(loc, dbox_list):
    """
    オフセット情報を使い、DBoxをBBoxに変換する。

    Parameters
    ----------
    loc:  [8732,4]
        SSDモデルで推論するオフセット情報。
    dbox_list: [8732,4]
        DBoxの情報

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBoxの情報
    """

    # DBoxは[cx, cy, width, height]で格納されている
    # locも[Δcx, Δcy, Δwidth, Δheight]で格納されている

    # オフセット情報からBBoxを求める
    boxes = torch.cat(
        (
            dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
            dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2),
        ),
        dim=1,
    )
    # boxesのサイズはtorch.Size([8732, 4])となります

    # BBoxの座標情報を[cx, cy, width, height]から[xmin, ymin, xmax, ymax] に
    boxes[:, :2] -= boxes[:, 2:] / 2  # 座標(xmin,ymin)へ変換
    boxes[:, 2:] += boxes[:, :2]  # 座標(xmax,ymax)へ変換

    return boxes


# Non-Maximum Suppressionを行う関数


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppressionを行う関数。
    boxesのうち被り過ぎ（overlap以上）のBBoxを削除する。

    Parameters
    ----------
    boxes : [確信度閾値（0.01）を超えたBBox数,4]
        BBox情報。
    scores :[確信度閾値（0.01）を超えたBBox数]
        confの情報

    Returns
    -------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count：int
        nmsを通過したBBoxの数
    """

    # returnのひな形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep：torch.Size([確信度閾値を超えたBBox数])、要素は全部0

    # 各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # boxesをコピーする。後で、BBoxの被り度合いIOUの計算に使用する際のひな形として用意
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # socreを昇順に並び変える
    v, idx = scores.sort(0)

    # 上位top_k個（200個）のBBoxのindexを取り出す（200個存在しない場合もある）
    idx = idx[-top_k:]

    # idxの要素数が0でない限りループする
    while idx.numel() > 0:
        i = idx[-1]  # 現在のconf最大のindexをiに

        # keepの現在の最後にconf最大のindexを格納する
        # このindexのBBoxと被りが大きいBBoxをこれから消去する
        keep[count] = i
        count += 1

        # 最後のBBoxになった場合は、ループを抜ける
        if idx.size(0) == 1:
            break

        # 現在のconf最大のindexをkeepに格納したので、idxをひとつ減らす
        idx = idx[:-1]

        # -------------------
        # これからkeepに格納したBBoxと被りの大きいBBoxを抽出して除去する
        # -------------------
        # ひとつ減らしたidxまでのBBoxを、outに指定した変数として作成する
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # すべてのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # wとhのテンソルサイズをindexを1つ減らしたものにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clampした状態でのBBoxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さが負になっているものは0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clampされた状態での面積を求める
        inter = tmp_w * tmp_h

        # IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
        rem_areas = torch.index_select(area, 0, idx)  # 各BBoxの元の面積
        union = (rem_areas - inter) + area[i]  # 2つのエリアの和（OR）の面積
        IoU = inter / union

        # IoUがoverlapより小さいidxのみを残す
        idx = idx[
            IoU.le(overlap)
        ]  # leはLess than or Equal toの処理をする演算です
        # IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に対してBBoxを囲んでいるため消去

    # whileのループが抜けたら終了

    return keep, count


# SSDの推論時にconfとlocの出力から、被りを除去したBBoxを出力する


class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(
            dim=-1
        )  # confをソフトマックス関数で正規化するために用意
        self.conf_thresh = (
            conf_thresh  # confがconf_thresh=0.01より高いDBoxのみを扱う
        )
        self.top_k = top_k  # nm_supressionでconfの高いtop_k個を計算に使用する, top_k = 200
        self.nms_thresh = nms_thresh  # nm_supressionでIOUがnms_thresh=0.45より大きいと、同一物体へのBBoxとみなす

    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝搬の計算を実行する。

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            オフセット情報。
        conf_data: [batch_num, 8732,num_classes]
            検出の確信度。
        dbox_list: [8732,4]
            DBoxの情報

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            （batch_num、クラス、confのtop200、BBoxの情報）
        """

        # 各サイズを取得
        num_batch = loc_data.size(0)  # ミニバッチのサイズ
        num_dbox = loc_data.size(1)  # DBoxの数 = 8732
        num_classes = conf_data.size(2)  # クラス数 = 21

        # confはソフトマックスを適用して正規化する
        conf_data = self.softmax(conf_data)

        # 出力の型を作成する。テンソルサイズは[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # cof_dataを[batch_num,8732,num_classes]から[batch_num, num_classes,8732]に順番変更
        conf_preds = conf_data.transpose(2, 1)

        # ミニバッチごとのループ
        for i in range(num_batch):

            # 1. locとDBoxから修正したBBox [xmin, ymin, xmax, ymax] を求める
            decoded_boxes = decode(loc_data[i], dbox_list)

            # confのコピーを作成
            conf_scores = conf_preds[i].clone()

            # 画像クラスごとのループ（背景クラスのindexである0は計算せず、index=1から）
            for cl in range(1, num_classes):

                # 2.confの閾値を超えたBBoxを取り出す
                # confの閾値を超えているかのマスクを作成し、
                # 閾値を超えたconfのインデックスをc_maskとして取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # gtはGreater thanのこと。gtにより閾値を超えたものが1に、以下が0になる
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # scoresはtorch.Size([閾値を超えたBBox数])
                scores = conf_scores[cl][c_mask]

                # 閾値を超えたconfがない場合、つまりscores=[]のときは、何もしない
                if scores.nelement() == 0:  # nelementで要素数の合計を求める
                    continue

                # c_maskを、decoded_boxesに適用できるようにサイズを変更します
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_maskをdecoded_boxesに適応します
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]で1次元になってしまうので、
                # viewで（閾値を超えたBBox数, 4）サイズに変形しなおす

                # 3. Non-Maximum Suppressionを実施し、被っているBBoxを取り除く
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k
                )
                # ids：confの降順にNon-Maximum Suppressionを通過したindexが格納
                # count：Non-Maximum Suppressionを通過したBBoxの数

                # outputにNon-Maximum Suppressionを抜けた結果を格納
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
                )

        return output  # torch.Size([1, 21, 200, 5])


# SSDクラスを作成する


class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inferenceを指定
        self.num_classes = cfg["num_classes"]  # クラス数=21

        # SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"]
        )

        # DBox作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論時はクラス「Detect」を用意します
        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        sources = list()  # locとconfへの入力source1～6を格納
        loc = list()  # locの出力を格納
        conf = list()  # confの出力を格納

        # vggのconv4_3まで計算する
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3の出力をL2Normに入力し、source1を作成、sourcesに追加
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vggを最後まで計算し、source2を作成、sourcesに追加
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # extrasのconvとReLUを計算
        # source3～6を、sourcesに追加
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # conv→ReLU→cov→ReLUをしたらsourceに入れる
                sources.append(x)

        # source1～6に、それぞれ対応する畳み込みを1回ずつ適用する
        # zipでforループの複数のリストの要素を取得
        # source1～6まであるので、6回ループが回る
        for x, l, c in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # l(x)とc(x)で畳み込みを実行
            # l(x)とc(x)の出力サイズは[batch_num, 4*アスペクト比の種類数, featuremapの高さ, featuremap幅]
            # sourceによって、アスペクト比の種類数が異なり、面倒なので順番入れ替えて整える
            # permuteで要素の順番を入れ替え、
            # [minibatch数, featuremap数, featuremap数,4*アスペクト比の種類数]へ
            # （注釈）
            # torch.contiguous()はメモリ上で要素を連続的に配置し直す命令です。
            # あとでview関数を使用します。
            # このviewを行うためには、対象の変数がメモリ上で連続配置されている必要があります。

        # さらにlocとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 最後に出力する
        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":  # 推論時
            # クラス「Detect」のforwardを実行
            # 返り値のサイズは torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])

        else:  # 学習時
            return output
            # 返り値は(loc, conf, dbox_list)のタプル
