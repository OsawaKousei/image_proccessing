import os
import tarfile
import urllib.request
import zipfile

# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# フォルダ「weights」が存在しない場合は作成する
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

# MSCOCOの2014 Val images [41K/6GB]をダウンロード
# 6GBのダウンロードと解凍なので時間がかかります（10分弱）
url = "http://images.cocodataset.org/zips/val2014.zip"
target_path = os.path.join(data_dir, "val2014.zip")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

    zip = zipfile.ZipFile(target_path)
    zip.extractall(data_dir)  # ZIPを解凍
    zip.close()  # ZIPファイルをクローズ
