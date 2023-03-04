import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import torch 
import imageio 
import cv2 as cv 
from torch.utils.data import Dataset, dataloader, random_split 
import torch.nn.functional as F 
import torch.nn as nn
from torchvision import transforms


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
        self.data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
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

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class MyResNet(nn.Module):
    def __init__(self, in_channels, out_chanels, used_1_1_conv=False, stride=1):
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_chanels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_chanels, out_chanels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chanels)
        self.bn2 = nn.BatchNorm2d(out_chanels)
        self.relu = nn.ReLU()
        if used_1_1_conv:
            self.conv3 = nn.Conv2d(in_channels, out_chanels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x))) 
        y = self.bn2(self.conv2(y)) 
        if self.conv3:
            x = self.conv3(x)
        return self.relu(y+x) 

class MyRes(nn.Module):
    def __init__(self):
        super(MyRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            # 每个resnet有两个conv
            MyResNet(64, 64, 1), 
            MyResNet(64, 128, True, 2))
        self.conv1 = MyResNet(128, 256, True, 2)
        self.conv2 = MyResNet(256, 512, True, 2)
        self.conv3 = MyResNet(512, 1024, True, 2)
        self.yolo3_out = MyResNet(1024, 5, True, 1)
        self.yolo3_out_cat = MyResNet(1024, 256, True, 1)
        self.upsample = Upsample(scale_factor=2)
        self.yolo2_out_cat_out = MyResNet(768, 512, True, 1)
        self.yolo2_out = MyResNet(512, 5, True, 1)
        self.yolo2_out_cat = MyResNet(512, 128, True, 1)
        self.yolo1_out = MyResNet(384, 5, True, 1)
        

    def forward(self, x):
        out = self.conv(x)
        yolo1_in = self.conv1(out)
        yolo2_in = self.conv2(yolo1_in)
        yolo3_in = self.conv3(yolo2_in)
        yolo3_out = self.yolo3_out(yolo3_in)
        yolo2_cat = self.upsample(self.yolo3_out_cat(yolo3_in))
        yolo2_in = torch.cat([yolo2_in, yolo2_cat],1)
        yolo2_out_cat_out = self.yolo2_out_cat_out(yolo2_in)
        yolo2_out = self.yolo2_out(yolo2_out_cat_out)
        yolo1_cat = self.upsample(self.yolo2_out_cat(yolo2_out_cat_out))
        yolo1_in = torch.cat([yolo1_in, yolo1_cat],1)
        yolo1_out = self.yolo1_out(yolo1_in)
        return yolo1_out, yolo2_out, yolo3_out
