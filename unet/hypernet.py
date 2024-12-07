import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *
from utils.utils import fft_my, ifft_my
from .unet_model import DataConsistency_M, fft_my_2c, ifft_my_2c, DataConsistency_K
class hyperNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(hyperNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownTriConv(2, 48)
        self.down2 = DownTriConv(48, 64)
        self.down3 = DownTriConv(64, 128)
        self.down4 = DownTriConv(128, 256)
        self.up1 = UpTri(384, 128, bilinear)
        self.up2 = UpTri(192, 64, bilinear)
        self.up3 = UpTri(112, 48, bilinear)
        self.outc1 = OutConv(48, n_classes)

        self.down1k = DownTriConv(2, 48)
        self.down2k = DownTriConv(48, 64)
        self.down3k = DownTriConv(64, 128)
        self.down4k = DownTriConv(128, 256)
        self.up1k = UpTri(384, 128, bilinear)
        self.up2k = UpTri(192, 64, bilinear)
        self.up3k = UpTri(112, 48, bilinear)
        self.outc2 = OutConv(48, n_classes)

    def forward(self, img, x_k, mask):
        x1 = self.inc(x_k)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        pre_k = self.outc1(x)
        out1 = DataConsistency_K(pre_k,x_k,mask)
        pred_img1 = ifft_my(out1,(-2,-1))

        x2 = self.down1k(pred_img1)
        x3 = self.down2k(x2)
        x4 = self.down3k(x3)
        x5 = self.down4k(x4)
        x = self.up1k(x5, x4)
        x = self.up2k(x, x3)
        x = self.up3k(x, x2)
        pred_img2 = self.outc2(x)
        pred_img2 = DataConsistency_M(pred_img2,x_k,mask)

        return [pred_img2,pred_img1]