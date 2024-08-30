import os
import urllib.request
import zipfile
import tarfile

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml

# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


# mnist = fetch_openml('mnist_784', version=1, data_home="./data/")  # data_homeは保存先を指定します
mnist = fetch_openml(
    "mnist_784", version=1, data_home="./data/", as_frame=False
)
# Issue #153 2020年12月にリリースされたsklearn 0.24.0以降の仕様変更に合わせる場合

# データの取り出し
X = mnist.data
y = mnist.target

# フォルダ「data」の下にフォルダ「img_78」を作成する
data_dir_path = "./data/img_78/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# MNISTから数字7、8の画像だけフォルダ「img_78」に画像として保存していく
count7 = 0
count8 = 0
max_num = 200  # 画像は200枚ずつ作成する

for i in range(len(X)):

    # 画像7の作成
    if (y[i] == "7") and (count7 < max_num):
        file_path = "./data/img_78/img_7_" + str(count7) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count7 += 1

    # 画像8の作成
    if (y[i] == "8") and (count8 < max_num):
        file_path = "./data/img_78/img_8_" + str(count8) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28*28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count8 += 1

    # 7と8を200枚ずつ作成したらbreak
    if (count7 >= max_num) and (count8 >= max_num):
        break

# フォルダ「data」の下にフォルダ「test」を作成する
data_dir_path = "./data/test/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# 上記で7,8の画像を作成するのに使用したindexの最終値
i_start = i + 1
print(i_start)

# MNISTから数字7、8の画像だけフォルダ「img_78」に画像として保存していく
count2 = 0
count7 = 0
count8 = 0
max_num = 5  # 画像は5枚ずつ作成する

for i in range(i_start, len(X)):  # i_startから始める

    # 画像2の作成
    if (y[i] == "2") and (count2 < max_num):
        file_path = "./data/test/img_2_" + str(count2) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count2 += 1

    # 画像7の作成
    if (y[i] == "7") and (count7 < max_num):
        file_path = "./data/test/img_7_" + str(count7) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count7 += 1

    # 画像8の作成
    if (y[i] == "8") and (count8 < max_num):
        file_path = "./data/test/img_8_" + str(count8) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28*28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count8 += 1

# フォルダ「data」の下にフォルダ「img_78_28size」を作成する
data_dir_path = "./data/img_78_28size/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# MNISTから数字7、8の画像だけフォルダ「img_78_28size」に画像として保存していく
count7 = 0
count8 = 0
max_num = 200  # 画像は200枚ずつ作成する

for i in range(len(X)):

    # 画像7の作成
    if (y[i] == "7") and (count7 < max_num):
        file_path = "./data/img_78_28size/img_7_" + str(count7) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f.save(file_path)  # 保存
        count7 += 1

    # 画像8の作成
    if (y[i] == "8") and (count8 < max_num):
        file_path = "./data/img_78_28size/img_8_" + str(count8) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28*28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f.save(file_path)  # 保存
        count8 += 1

    if (count7 >= max_num) and (count8 >= max_num):
        break

# フォルダ「data」の下にフォルダ「test」を作成する
data_dir_path = "./data/test_28size/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# 上記で7,8の画像を作成するのに使用したindexの最終値
i_start = i + 1
print(i_start)

# MNISTから数字7、8の画像だけフォルダ「img_78」に画像として保存していく
count2 = 0
count7 = 0
count8 = 0
max_num = 5  # 画像は5枚ずつ作成する

for i in range(i_start, len(X)):  # i_startから始める

    # 画像2の作成
    if (y[i] == "2") and (count2 < max_num):
        file_path = "./data/test_28size/img_2_" + str(count2) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f.save(file_path)  # 保存
        count2 += 1

    # 画像7の作成
    if (y[i] == "7") and (count7 < max_num):
        file_path = "./data/test_28size/img_7_" + str(count7) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f.save(file_path)  # 保存
        count7 += 1

    # 画像8の作成
    if (y[i] == "8") and (count8 < max_num):
        file_path = "./data/test_28size/img_8_" + str(count8) + ".jpg"
        im_f = X[i].reshape(28, 28)  # 画像を28*28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f.save(file_path)  # 保存
        count8 += 1
