import numpy as np 
import torch 
import torch.nn as nn 
import torchvision 
from PIL import Image 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader, random_split


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
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        # 为了避免完全不相交的情况
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def nms(boxes, scores, iou_threshold=0.3):
    keep = []
    idxs = scores.argsort()
    while len(idxs) > 0:
        max_score_idx = idxs[-1]
        max_score_box = boxes[max_score_idx][None, :]
        keep.append(max_score_idx.item())
        if len(idxs) == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = bbox_iou(max_score_box, other_boxes)
        idxs = idxs[ious <= iou_threshold]
    return keep


#Darknet层
class DarknetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bnorm = True, leaky = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False if bnorm else True)
        self.bnorm = nn.BatchNorm2d(out_channels, eps = 1e-3) if bnorm else None
        self.leaky = nn.LeakyReLU(0.1) if leaky else None
    def forward(self, x):
        x = self.conv(x)
        if self.bnorm is not None:
            x = self.bnorm(x)
        if self.leaky is not None:
            x = self.leaky(x)
        return x

#DarkNet块       
class DarknetBlock(nn.Module):
    def __init__(self, layers, skip = True):
        super().__init__()
        self.skip = skip
        self.layers = nn.ModuleDict()
        for i in range(len(layers)):
            self.layers[layers[i]['id']] = DarknetLayer(layers[i]['in_channels'], layers[i]['out_channels'], layers[i]['kernel_size'],
                                                 layers[i]['stride'], layers[i]['padding'], layers[i]['bnorm'],
                                                 layers[i]['leaky'])
    def forward(self, x):
        count = 0
        for _, layer in self.layers.items():
            if count == (len(self.layers) - 2) and self.skip:
                skip_connection = x
            count += 1
            x = layer(x)
        return x + skip_connection if self.skip else x

#DarkNet网络
class Yolov3(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False)
        #layer0 -> layer4, input = (3, 416, 416), flow_out = (64, 208, 208)
        self.blocks = nn.ModuleDict()
        self.blocks['block0_4'] = DarknetBlock([
            {'id': 'layer_0', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_1', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_2', 'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_3', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer5 -> layer8, input = (64, 208, 208), flow_out = (128, 104, 104)
        self.blocks['block5_8'] = DarknetBlock([
            {'id': 'layer_5', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_6', 'in_channels': 128, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_7', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer9 -> layer11, input = (128, 104, 104), flow_out = (128, 104, 104)
        self.blocks['block9_11'] = DarknetBlock([
            {'id': 'layer_9', 'in_channels': 128, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_10', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer12 -> layer15, input = (128, 104, 104), flow_out = (256, 52, 52)
        self.blocks['block12_15'] = DarknetBlock([
            {'id': 'layer_12', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_13', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_14', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer16 -> layer36, input = (256, 52, 52), flow_out = (256, 52, 52)
        self.blocks['block16_18'] = DarknetBlock([
            {'id': 'layer_16', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_17', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block19_21'] = DarknetBlock([
            {'id': 'layer_19', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_20', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block22_24'] = DarknetBlock([
            {'id': 'layer_22', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_23', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block25_27'] = DarknetBlock([
            {'id': 'layer_25', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_26', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block28_30'] = DarknetBlock([
            {'id': 'layer_28', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_29', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block31_33'] = DarknetBlock([
            {'id': 'layer_31', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_32', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block34_36'] = DarknetBlock([
            {'id': 'layer_34', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_35', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer37 -> layer40, input = (256, 52, 52), flow_out = (512, 26, 26)
        self.blocks['block37_40'] = DarknetBlock([
            {'id': 'layer_37', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_38', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_39', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer41 -> layer61, input = (512, 26, 26), flow_out = (512, 26, 26)
        self.blocks['block41_43'] = DarknetBlock([
            {'id': 'layer_41', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_42', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block44_46'] = DarknetBlock([
            {'id': 'layer_44', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_45', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block47_49'] = DarknetBlock([
            {'id': 'layer_47', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_48', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block50_52'] = DarknetBlock([
            {'id': 'layer_50', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_51', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block53_55'] = DarknetBlock([
            {'id': 'layer_53', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_54', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block56_58'] = DarknetBlock([
            {'id': 'layer_56', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_57', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block59_61'] = DarknetBlock([
            {'id': 'layer_59', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_60', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer62 -> layer65, input = (512, 26, 26), flow_out = (1024, 13, 13)
        self.blocks['block62_65'] = DarknetBlock([
            {'id': 'layer_62', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_63', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_64', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer66 -> layer74, input = (1024, 13, 13), flow_out = (1024, 13, 13)
        self.blocks['block66_68'] = DarknetBlock([
            {'id': 'layer_66', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_67', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block69_71'] = DarknetBlock([
            {'id': 'layer_69', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_70', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block72_74'] = DarknetBlock([
            {'id': 'layer_72', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_73', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer75 -> layer79, input = (1024, 13, 13), flow_out = (512, 13, 13)
        self.blocks['block75_79'] = DarknetBlock([
            {'id': 'layer_75', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_76', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_77', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_78', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_79', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer80 -> layer82, input = (512, 13, 13), yolo_out = (255, 13, 13)
        self.blocks['yolo_82'] = DarknetBlock([
            {'id': 'layer_80', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_81', 'in_channels': 1024, 'out_channels': 255, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': False, 'leaky': False}
        ], skip = False)
        #layer83 -> layer86, input = (512, 13, 13), -> (256, 13, 13) -> upsample and concate layer61(512, 26, 26), flow_out = (768, 26, 26)
        self.blocks['block83_86'] = DarknetBlock([
            {'id': 'layer_84', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer87 -> layer91, input = (768, 26, 26), flow_out = (256, 26, 26)
        self.blocks['block87_91'] = DarknetBlock([
            {'id': 'layer_87', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_88', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_89', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_90', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_91', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer92 -> layer94, input = (256, 26, 26), yolo_out = (255, 26, 26)
        self.blocks['yolo_94'] = DarknetBlock([
            {'id': 'layer_92', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_93', 'in_channels': 512, 'out_channels': 255, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': False, 'leaky': False}
        ], skip = False)
        #layer95 -> layer98, input = (256, 26, 26), -> (128, 26, 26) -> upsample and concate layer36(256, 52, 52), flow_out = (384, 52, 52)
        self.blocks['block95_98'] = DarknetBlock([
            {'id': 'layer_96', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer99 -> layer106, input = (384, 52, 52), yolo_out = (255, 52, 52)
        self.blocks['yolo_106'] = DarknetBlock([
            {'id': 'layer_99', 'in_channels': 384, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_100', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_101', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_102', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_103', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_104', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_105', 'in_channels': 256, 'out_channels': 255, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': False, 'leaky': False}
        ], skip = False)
        
    def forward(self, x):
        x = self.blocks['block0_4'](x)
        x = self.blocks['block5_8'](x)
        x = self.blocks['block9_11'](x)
        x = self.blocks['block12_15'](x)
        x = self.blocks['block16_18'](x)
        x = self.blocks['block19_21'](x)
        x = self.blocks['block22_24'](x)
        x = self.blocks['block25_27'](x)
        x = self.blocks['block28_30'](x)
        x = self.blocks['block31_33'](x)
        x = self.blocks['block34_36'](x)
        skip36 = x
        x = self.blocks['block37_40'](x)
        x = self.blocks['block41_43'](x)
        x = self.blocks['block44_46'](x)
        x = self.blocks['block47_49'](x)
        x = self.blocks['block50_52'](x)
        x = self.blocks['block53_55'](x)
        x = self.blocks['block56_58'](x)
        x = self.blocks['block59_61'](x)
        skip61 = x
        x = self.blocks['block62_65'](x)
        x = self.blocks['block66_68'](x)
        x = self.blocks['block69_71'](x)
        x = self.blocks['block72_74'](x)
        x = self.blocks['block75_79'](x)
        yolo_82 = self.blocks['yolo_82'](x)
        x = self.blocks['block83_86'](x)
        x = self.upsample(x)
        x = torch.cat((x, skip61), dim = 1)
        x = self.blocks['block87_91'](x)
        yolo_94 = self.blocks['yolo_94'](x)
        x = self.blocks['block95_98'](x)
        x = self.upsample(x)
        x = torch.cat((x, skip36), dim = 1)
        yolo_106 = self.blocks['yolo_106'](x)
        return yolo_82, yolo_94, yolo_106  
      
      
model = Yolov3()
data = np.random.random(size=(1, 3, 416, 416))
data = torch.tensor(data).to(torch.float32)
yolo1, yolo2, yolo3 = model(data)
print(yolo1.size(), yolo2.size(), yolo3.size())
data = np.random.random(size=(1, 3, 416, 416))
data = torch.tensor(data).to(torch.float32)
yolo1, yolo2, yolo3 = model(data)
print(yolo1.size(), yolo2.size(), yolo3.size())
plt.imshow(torch.squeeze(img, 0).permute(1,2,0).detach().numpy())
box = []
for t in range(3):
    _, grid_w, grid_h = y_hat[t][0].shape
    per_y = y_hat[t][0].permute(1,2,0).reshape((grid_w, grid_h, 3, 85)).detach().numpy()
    for i in range(grid_w):
        for j in range(grid_h):
            for k in range(3):
                for v in range(80):
                    if _sigmoid(per_y[i, j, k, 4])*_sigmoid(per_y[i, j, k, 5+v]) > 0.75:
                        bx = (_sigmoid(per_y[i, j, k, 0]) + j) * grid_w
                        by = (_sigmoid(per_y[i, j, k, 1]) + i) * grid_h 
                        bw = anchors[t][2*k]*np.exp(per_y[i, j, k, 2])
                        bh = anchors[t][2*k+1]*np.exp(per_y[i, j, k, 3])
                        box.append((bx, by, bw, bh))
data = Image.open(r"/home/tao_wei/remote/target_detection/zbra.jpg")
data = data.resize((416,416))
plt.imshow(data)
ax = plt.gca()
for i in range(len(box)):
    y1, x1 = box[i][0], box[i][1]
    width, height = box[i][2], box[i][3]
    rect = plt.Rectangle((x1, y1), width, height, fill=False, color='white')
    ax.add_patch(rect)
plt.show()
