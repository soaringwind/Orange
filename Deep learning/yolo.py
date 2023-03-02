import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import torch 
import imageio 
import cv2 as cv 
from torch.utils.data import Dataset, dataloader, random_split 
import torch.nn.functional as F 
import torch.nn as nn


def plot_react_and_show(image_array, size):
    for i in range(len(size)):
        top_left_x, top_left_y, width, height = size[i]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor = 'red',linewidth=1)
        ax.add_patch(rect)
    plt.imshow(image_array) # 图像数组
    plt.show()

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def letterbox(img, pad_value, size):
    img = np.transpose(img, [2, 0, 1])
    img = torch.from_numpy(img)
    img, pad = pad_to_square(img, pad_value)
    img = resize(img, size)
    return img, pad 



whole_img_num = 0
with open("/home/tao_wei/remote/face_detection/yolo.txt", "w") as wp:
    for i in range(1, 10):
        with open("/home/tao_wei/remote/face_detection/FDDB-folds/FDDB-fold-0%s-ellipseList.txt"%i) as fp:
            while True:
                path = fp.readline().strip("\n")
                if not path:
                    break
                file_path = "/home/tao_wei/remote/face_detection/FDDB-face/" + path + ".jpg"
                img = imageio.imread(file_path)
                num = int(fp.readline())
                for _ in range(num):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, _, _ = fp.readline().split(' ')
                    x = int(float(center_x) - float(minor_axis_radius))
                    y = int(float(center_y) - float(major_axis_radius))
                    w = int(2*float(minor_axis_radius))
                    h = int(2*float(major_axis_radius))
                    if x+w > img.shape[0] or y+h > img.shape[1] or y < 0 or x < 0:
                        continue
                    whole_img_num += 1
                    wp.write("%s %d %d %d %d\n"%(file_path, x, y, w, h))
                # plot_react_and_show(img, [[x, y, w, h]])
print("总数据量为%s"%whole_img_num)



class YoloDataset(Dataset):
    def __init__(self, path, img_size=416, augment=False) -> None:
        self.img_size = img_size
        self.augment = augment
        with open(path, 'r') as file:
            img_files = file.read().splitlines()
            self.img_files = list(filter(lambda x:len(x)>0, img_files))
    
    def __len__(self):
        return len(self.img_size)
    
    def __getitem__(self, index):
        img_path, x, y, w, h = self.img_files[index].split(" ")
        x, y, w, h = int(x), int(y), int(w), int(h)
        img = imageio.imread(img_path)
        img_h, img_w, _ = img.shape
        img, pad = letterbox(img, 0, self.img_size)
        ratio = self.img_size / max(img_h, img_w)
        x, y, w, h = ratio*x+pad[0], ratio*y+pad[2], ratio*w, ratio*h 
        return {"data": img, "rect": [x, y, w, h]}
