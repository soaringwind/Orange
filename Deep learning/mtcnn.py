import imageio 
import os 
import torch 
import imgaug as ia 
import torch.nn as nn 
import numpy as np 
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    得到bbox的坐标
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def nms(boxes, scores, iou_threshold=0.3):
    keep = []
    idxs = scores.argsort()
    while len(idxs) > 0:
        max_score_idx = idxs[-1]
        max_score_box = boxes[max_score_idx][None, :]
        keep.append(max_score_idx)
        if len(idxs) == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = bbox_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]
    keep = idxs.new(keep)
    return keep
    
    
   anno_src = r"/home/tao_wei/remote/face_detection/CelebA/Anno/list_bbox_celeba.txt"
img_dir = r""
save_path = r"/home/tao_wei/remote/face_detection/CelebA/MTCNN/dataSet/"
float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
def gen_sample(face_size, stop_value):
    print("generate size: %s"%face_size)
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")
    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")
    positive_count = 0
    negative_count = 0
    part_count = 0
    with open(positive_anno_filename, "w") as op:
        with open(negative_anno_filename, "w") as ep:
            with open(part_anno_filename, "w") as ap:
                with open(anno_src, "r") as fp:
                    for i, line in enumerate(fp.readlines()):
                        if i < 2:
                            continue
                        print(i, line.strip().split())
                        img_filename, x, y, w, h = line.strip().split()
                        image_file = os.path.join(img_dir, img_filename)
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        if x < 0 or y < 0 or w < 0 or h < 0:
                            continue
                        img = imageio.imread(image_file)
                        img_w, img_h = img.shape
                        if x+w > img_w or y+h > img_h:
                            continue
                        boxes = torch.tensor([[x, y, x+w, y+h]])
                        side_len = max(w, h)
                        seed = float_num[np.random.randint(0, len(float_num))]
                        count = 0
                        for _ in range(4):
                            new_side_len = int(side_len + np.random.randint(int(-side_len*seed), int(side_len*seed)))
                            new_x = int(x + np.random.randint(int(-x*seed), int(x*seed)))
                            new_y = int(y + np.random.randint(int(-y*seed), int(y*seed)))
                            if new_x < 0 or new_y < 0 or new_x+new_side_len > img_w or new_y+new_side_len > img_h:
                                continue
                            offset_x1 = (x-new_x) / new_side_len
                            offset_y1 = (y-new_y) / new_side_len
                            offset_x2 = (x+w-new_x-new_side_len) / new_side_len
                            offset_y2 = (y+h-new_y-new_side_len) / new_side_len
                            new_boxes = torch.tensor([[new_x, new_y, new_x+new_side_len, new_y+new_side_len]])
                            scale = ia.augmenters.Scale(face_size)
                            face_img = img[new_x:new_x+new_side_len, new_y:new_y+new_side_len]
                            face_img = scale(image=face_img)
                            iou = bbox_iou(boxes, new_boxes)
                            if iou >= 0.65:
                                op.write(
                                    "positive/%s.jpg %s %s %s %s %s"%(positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2)
                                )
                                op.flush()
                                imageio.imwrite(os.path.join(positive_image_dir, "%s.jpg"%positive_count))
                                positive_count += 1
                            elif 0.4 <= iou < 0.65:
                                ap.write("part/%s.jpg %s %s %s %s %s"%(part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2))
                                ap.flush()
                                imageio.imwrite(os.path.join(part_image_dir, "%s.jpg"%part_count))
                                part_count += 1
                            elif iou < 0.1:
                                fp.write("negative/%s.jpg %s %s %s %s %s"%(negative_count, 0, 0, 0, 0, 0))
                                fp.flush()
                                imageio.imwrite(os.path.join(negative_image_dir, "%s.jpg"%(negative_count)))
                                negative_count += 1
                            count += 1
                        if count >= stop_value:
                            break
gen_sample(12, 1)

class PNet(nn.Module): 
    def __init__(self):
        super(PNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1), 
            nn.BatchNorm2d(10), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=1, stride=1), 
        )

    def forward(self, x):
        feature = self.conv(x)
        out = feature.view((x.shape[0], -1))
        category = out[:, 0:2]
        offset = out[:, 2:]
        return category, offset

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1), 
            nn.BatchNorm2d(28), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1), 
            nn.BatchNorm2d(48), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(48, out_channels=64, kernel_size=2, stride=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
        )
        self.fc = nn.Sequential(
            nn.Linear(3*3*64, 128), 
            nn.ReLU(), 
            nn.Linear(128, 6)
        )

    def forward(self, x):
        feature = self.conv(x)
        out = self.fc(feature.view(x.shape[0], -1))
        category = out[:, 0:2]
        offset = out[:, 2:]
        return category, offset

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),  
        )

        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256), 
            nn.ReLU(), 
            nn.Linear(256, 6), 
        )

    def forward(self, x):
        feature = self.conv(x)
        out = self.fc(feature.view(x.shape[0], -1))
        category = out[:, 0:2]
        offset = out[:, 2:]
        return category, offset
        
class FaceDataset(Dataset):
    def __init__(self, data, label, offset, transform):
        self.data = data
        self.label = label
        self.offset = offset
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        img = self.data[idx]
        label = self.label[idx]
        offset = self.offset[idx]
        if self.transform:
            img = self.transform(img)
        return {"data": img, "label": label, "offset": offset}

    def __len__(self):
        return len(self.label)
        
tfs = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5,], [0.5, ])
])
data = None
label = None
dataset = FaceDataset(data, label, tfs)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


class Trainer(object):
    def __init__(self, net, dataloader):
        self.net = net
        self.net.train()
        self.dataloader = dataloader
        self.cls_loss = nn.CrossEntropyLoss()
        self.offset_loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train(self, epoch_num=10):
        for epoch in range(epoch_num):
            running_loss = 0
            for i, data in enumerate(self.dataloader, 0):
                img_data = data["data"]
                img_label = data["label"]
                img_offset = data["offset"]
                out_category, out_offset = self.net(img_data)
                # 只衡量正负样本的损失
                category_mask = torch.lt(img_label, 2)
                category = torch.masked_select(img_label, category_mask)
                out_category = torch.masked_select(out_category, category_mask)
                cls_loss_val = self.cls_loss(out_category, category)
                
                offset_mask = torch.gt(img_label, 0)
                offset = torch.masked_select(img_offset, offset_mask)
                out_offset = torch.masked_select(out_offset, offset_mask)
                offset_loss_val = self.offset_loss(offset, out_offset)

                loss = cls_loss_val + offset_loss_val
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("epoch: %s, loss: %s"%(epoch, running_loss/(i+1)))


Ptrainer = Trainer(PNet(), dataloader)
Ptrainer.train(1)
