from sklearn import cluster
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import sys
import shutil
import argparse


def load_image(file_path):
    
    file_list = glob.glob(os.path.join(file_path,"*"))
    img_list = []
    img_file_list = []
    for file_name in file_list:
        file_name = file_name.replace("\\", "/")
        # imgは[y][x][c]
        try:
            img = cv2.imread(file_name)
            img = cv2.resize(img, (224, 224))
            img = img.swapaxes(0,1)
            print("{}を読み込みました".format(file_name))
            img_list.append(img)
            img_file_list.append(file_name)
        except:
            pass
    
    # 画像データは[枚数][画像](画像は1次元に直す)
    data_array = np.array(img_list).reshape(len(img_list), -1).astype(np.float64)
    return data_array, img_file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="指定されたディレクトリの画像を分類する")
    parser.add_argument("-i", "--input", help="分類する画像のあるディレクトリ", required=True)
    parser.add_argument("-o", "--output", help="分類した画像の出力先ディレクトリ", required=True)
    parser.add_argument("-n", "--num", help="分類する数。デフォルト=8", type=int, default=8)
    
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output
    div_num = args.num
        
    data, file_list = load_image(input_dir)
    print("学習を開始します(cluster={})".format(div_num))
    model = cluster.KMeans(n_clusters=div_num)
    model.fit(data)
    print("学習を終了しました")
    labels = model.labels_
    
    for i, label in enumerate(labels):
        output_path = os.path.join(output_dir, str(label))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_name = file_list[i].split("/")[-1]
        output_file = os.path.join(output_path, img_name)
        shutil.copyfile(file_list[i], output_file)
        print("{}を{}にコピーしました".format(file_list[i], output_file))
        
        
