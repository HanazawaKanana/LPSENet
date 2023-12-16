import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU

##SimAM
class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

##添加的模块，尝试替换1024-》256的卷积层，分别对四层特征进行卷积
class LightConv(nn.Module):
    def __init__(self,in_channels):
        super(LightConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels//2)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=3, groups= in_channels//2, bias=False, dilation=3)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x=self.bn(self.conv1x1(self.conv1(x)))
        x=self.bn(self.conv2(x))
        return self.relu(x)

class PSENet(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.lightconv=LightConv(256)##轻量级的卷积部分
        self.sm=simam_module(256)##定义simam层

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training :##and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)
        if not self.training:## and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # FPN
        f1, f2, f3, f4 = self.fpn(f[0], f[1], f[2], f[3])
        
        ##轻量级卷积结构
        ##f1 = self.lightconv(f1)##sm(f1)
        ##f2 = self.lightconv(f2)##sm(f2)
        ##f3 = self.lightconv(f3)##sm(f3)
        ##f4 = self.lightconv(f4)##sm(f4)

        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training:## and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection

        det_out = self.det_head(f)

        if not self.training:## and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 1)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs
