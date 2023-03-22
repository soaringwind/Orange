# https://github.com/eriklindernoren/PyTorch-YOLOv3

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
mydataset = YoloDataset("/home/tao_wei/remote/face_detection/yolo.txt")
img = mydataset[0]['data']
label = mydataset[0]['rect']
img = torch.unsqueeze(img, 0)
model = MyRes()
print(img.shape)
prediction = model(img)
print(prediction[0].shape)
# img = np.transpose(img, [1, 2, 0])
# plot_react_and_show(img, [mydataset[0]['rect']])


def build_targets(prediction, label):
    # label: id, x, y, w, h
    num_of_anchors, num_of_targets = 1, label.shape[0]
    tcls, tbox, indices, anch = [], [], [], [] 
    anchors = [[33,23], [59,119],[156,198]]
    stride = [416//13, 416//26, 416//52]
    anchors = torch.tensor(anchors).float().view(-1, 2)
    # 为了能够把targets的尺寸放大回去
    gain = torch.ones(6)
    anchor_index = torch.arange(num_of_anchors).float().view(num_of_anchors, 1).repeat(1, num_of_targets)
    anchor_index = anchor_index.view(-1, 1)
    # 最终得到的矩阵是num_of_anchors*num_of_targets*(len(label)+1)
    targets = torch.cat((label.repeat(num_of_anchors, 1, 1), anchor_index[:, :, None]), 2)
    for i in range(len(prediction)):
        select_anchor = anchors[i, :].view(-1,2)/stride[i]
        gain[1:5] = torch.tensor(prediction[i].shape)[[2, 1, 2, 1]]
        t = targets*gain
        if num_of_targets:
            r = t[:, :, 3:5] / select_anchor[:, None]
            j = torch.max(r, 1./r).max(2)[0] < 4 
            t = t[j]
        else:
            t = targets[0]
        b = t[:, 0].long()
        gxy = t[:, 1:3]
        gwh = t[:, 3:5]
        gij = gxy.long()
        gi, gj = gij.T
        a = t[:, 5].long()
        # 框的左上角位置
        indices.append((b, a, gj.clamp_(0, gain[2].long()-1), gi.clamp_(0, gain[1].long()-1)))
        # 记录补偿的位置
        tbox.append(torch.cat((gxy-gij, gwh), 1))
        # 记录正确的锚点框
        anch.append(anchors[a])
        # 记录类别
        tcls.append(1)
    return tcls, tbox, indices, anch

build_targets(prediction, label)


import math


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    box2 = box2.T 
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3] 
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3] 
    else:
        b1_x1, b1_x2 = box1[0]-box1[2]/2, box1[0]+box1[2]/2
        b1_y1, b1_y2 = box1[1]-box1[3]/2, box1[1]+box1[3]/2
        b2_x1, b2_x2 = box2[0]-box2[2]/2, box2[0]+box2[2]/2
        b2_y1, b2_y2 = box2[1]-box2[3]/2, box2[1]+box2[3]/2
    inter = (torch.min(b1_x2, b2_x2)-torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2)-torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2-b1_x1, b1_y2-b1_y1+eps 
    w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
    union = w1*h1+w2*h2-inter+eps 
    iou = inter / union 
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps 
            rho2 = ((b2_x1+b2_x2-b1_x1-b1_x2)**2+(b2_y1+b2_y2-b1_y1-b1_y2)**2)/4
            if DIoU:
                return iou-rho2/c2 
            elif CIoU:
                v = (4/math.pi**2)*torch.pow(torch.atan(w2/h2)-torch.atan(w1/h1), 2)
                with torch.no_grad():
                    alpha = v / ((1+eps)-iou+v)
                return iou - (rho2/c2+v*alpha)
        else:
            c_area = cw*ch + eps 
    else:
        return iou 
    
    
    def compute_loss(prediction, label):
    lcls, lbox, lobj = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    tcls, tbox, indices, anchors = build_targets(prediction, label)
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
    for layer_index, layer_predictions in enumerate(prediction):
        b, anchor, grid_j, grid_i = indices[layer_index]
        tobj = torch.zeros_like(layer_predictions[..., 0])
        num_targets = anchor.shape[0]
        if num_targets:
            ps = layer_predictions[b, grid_j, grid_i]
            pxy = ps[:, :2].sigmoid()
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            pbox = torch.cat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            lbox += (1.0-iou).mean()
            tobj[b, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)
            if ps.size(1)-5>1:
                t = torch.zeros_like(ps[:, 5:])
                t[range(num_targets), tcls[layer_index]] = 1
                lcls += BCEcls(ps[:, 5:], t)
        lobj += BCEobj(layer_predictions[..., 4], tobj)
    lbox *= 0.05
    lobj *= 1.0 
    lcls *= 0.5
    loss = lbox+lobj+lcls
    return loss, torch.cat((lbox, lobj, lcls, loss))



loss, loss_components = compute_loss(prediction, label)
# loss.backward()



def col_fn(batch):
    data = []
    rect = []
    num = 0
    for i, batch_data in enumerate(batch):
        if batch_data is None:
            continue
        data.append(batch_data["data"])
        batch_data["rect"][:, 0] = num
        rect.append(batch_data["rect"])
        num += 1
    return {"data": torch.stack(data), "rect": torch.stack(rect)}


# yolo整体流程
# 1. 构建训练、验证、测试数据集
train_data_len = int(0.7*len(mydataset))
valid_data_len = int(0.1*len(mydataset))
test_data_len = len(mydataset) - train_data_len - valid_data_len
train_data, valid_data, test_data = random_split(mydataset, [train_data_len, valid_data_len, test_data_len])
train_data_load = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=col_fn)
valid_data_load = DataLoader(valid_data, batch_size=256, shuffle=False)
test_data_load = DataLoader(test_data, batch_size=256, shuffle=False)
# 2. 选择优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 3. 开始训练
for epoch in range(5):
    print("-----------training--------------")
    running_loss = 0 
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_data_load, 0):
        outputs = model(data["data"])
        labels = data["rect"]
        loss, loss_components = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss
        if i % 10 == 0:
            print("当前 %s 次损失: %s"%(i, running_loss/(i+1)))
    print("epoch %s 次损失: %s"%(epoch+1, running_loss/(i+1)))


    
