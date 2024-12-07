""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import (
    DownTriConv,TriConv,DoubleConv,
    Conv,Down,UpTri,Up,OutConv
)
from utils.utils import fft_my, ifft_my
from .unet_model import (DataConsistency_M, DataConsistency_K)
from .modules import (
    ResidualConv,
    Conv,
    ASPP,
    ASPP_Down,
    ASPP_Res,
    ASPP_Res_Down,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
    AttentionWeightChannel,
    AttentionWeightChannelECA,
    ECA_layer
)
from .ScConv import ScConv

# class DataConsistency_M(nn.Module):
#     # DC layer
#     def __init__(self):
#         super(DataConsistency_M, self).__init__()

#     def forward(self, x, x_k, mask):
#         '''
#         input [n 2 w h]
#         operation img space[n 2 w h ] - > [n w h ](complex type)
#                   k space[n 2 w h ] - > [n w h ](complex type)
#         output [n 2 w h]
#         '''
#         gene_img = torch.complex(x[:,0,:,:],x[:,1,:,:])
#         gene_img = torch.squeeze(gene_img)
#         gene_k = torch.complex(x_k[:,0,:,:],x_k[:,1,:,:])
#         gene_k = torch.squeeze(gene_k)
#         output = torch.rand(size=gene_img.size(),dtype=torch.complex64).cuda() #[n w h]
#         # print("gene_img.size()",gene_img.size())
#         if len(gene_img.size())==2:
#             gene_img = torch.unsqueeze(gene_img,0)
#             gene_k = torch.unsqueeze(gene_k,0)
#             output = torch.unsqueeze(output,0)
#         for i in range(gene_img.size()[0]):
#             pred_k = fft_my(gene_img[i],(-2,-1)) * (1.0 - mask[i])
#             out_fft = gene_k[i] + pred_k
#             output[i] = ifft_my(out_fft,(-2,-1)) #[][n w h]
#         output_cat = torch.cat([torch.unsqueeze(output.real,1),torch.unsqueeze(output.imag,1)],dim=1)
#         # output_cat = torch.unsqueeze(output_cat,1)
#         return output_cat

# class DataConsistency_K(nn.Module):
#     # DC layer
#     def __init__(self):
#         super(DataConsistency_K, self).__init__()

#     def forward(self, x ,x_k, mask):
#         '''
#         input k [n 2 w h ] x_k [n 2 w h], mask [n w h]
#         operation k space[n 2 w h ] - > [n w h ](complex type)
#                   k space[n 2 w h ] - > [n w h ](complex type)
#         output [n 2 w h]
#         '''
#         # x, x_k, mask  = input[0],input[1],input[2]
#         gene_k = torch.complex(x[:,0,:,:],x[:,1,:,:])
#         gene_k = torch.squeeze(gene_k) #[n w h]
#         ori_k = torch.complex(x_k[:,0,:,:],x_k[:,1,:,:])
#         ori_k = torch.squeeze(ori_k) #[n w h ]
#         output = torch.rand(size=gene_k.size(),dtype=torch.complex64).cuda() #[n w h]
#         # print("gene_img.size()",gene_img.size())
#         for i in range(gene_k.size()[0]):
#             pred_k = ori_k[i] * (1.0 - mask[i])
#             out_fft = gene_k[i] + pred_k
#             # print("out_fft.shape",out_fft.shape)
#             output[i] = out_fft
#         output_cat = torch.cat([torch.unsqueeze(output.real,1),torch.unsqueeze(output.imag,1)],dim=1)
#         # output_cat = torch.unsqueeze(output_cat,1)
#         return output_cat


class UNetDC_short_first_rassp_ECA_ScConv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_rassp_ECA_ScConv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ScConv(256)
        self.awc2 = ScConv(64)
        self.awc3 = ScConv(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Res_Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        x = self.awc2(x)
        x = self.up4(x, x1) #64
        x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class UNetDC_short_first_rassp_ECAfront_ScConv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_rassp_ECAfront_ScConv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ScConv(256)
        self.awc2 = ScConv(64)
        self.awc3 = ScConv(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Res_Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x = self.awc1(x)
        x2 = self.down1(x1) #128
        x = self.awc2(x)
        x3 = self.down2(x2) #256
        x = self.awc3(x)
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #64

        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class UNetDC_short_ECA_rassp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_ECA_rassp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Res_Down(64, 128)
        self.down2 = ASPP_Res_Down(128, 256,rate=[1,2,3])
        self.down3 = ASPP_Res_Down(256, 512,rate=[1,2,3])
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        x = self.awc2(x)        
        x = self.up4(x, x1) #64
        x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class UNetDC_short_first_rassp_ECA_half(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_rassp_ECA_half, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(128)
        self.awc2 = ECA_layer(32)
        self.awc3 = ECA_layer(32)

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = ASPP_Res_Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(192, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(32, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        x = self.awc2(x)
        x = self.up4(x, x1) #64
        x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]

class UNetDC_short_first_rassp_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_rassp_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Res_Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        x = self.awc2(x)
        x = self.up4(x, x1) #64
        x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class UNetDC_short_mid_rassp_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_mid_rassp_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = ASPP_Res_Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        x = self.awc2(x)        
        x = self.up4(x, x1) #64
        x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]



class UNetDC_short_last_rassp_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_last_rassp_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = ASPP_Res_Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        x = self.awc2(x)        
        x = self.up4(x, x1) #64
        x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class UNetDC_short_first_rassp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_rassp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.awc1 = ECA_layer(256)
        # self.awc2 = ECA_layer(64)
        # self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Res_Down(64, 364)
        self.down2 = Down(364, 364)
        self.down3 = Down(364, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(876, 512, bilinear)
        self.up3 = Up(876, 364, bilinear)
        self.up4 = Up(428, 256, bilinear)
        self.dc = DataConsistency_M()
        self.outc1 = OutConv(256, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        # x = self.awc1(x)
        x = self.up3(x, x2) #64
        # x = self.awc2(x)
        x = self.up4(x, x1) #64
        # x = self.awc3(x)
        x = self.outc1(x)
        logits = self.outc(x)

        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class UNetDC_short_first_rassp_first_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_rassp_first_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Res_Down(64, 128)
        # self.down1 = Down(64, 128)        
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc1(x)
        x = self.up3(x, x2) #64
        # x = self.awc2(x)
        x = self.up4(x, x1) #64
        # x = self.awc3(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]