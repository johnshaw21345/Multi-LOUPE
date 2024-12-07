""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import (
    DownTriConv,TriConv,DoubleConv,
    Conv,Down,UpTri,Up,OutConv
)
from utils.utils import fft_my, ifft_my

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

def fft_my_2c(x):
    gene_img = torch.complex(x[:,0,:,:],x[:,1,:,:])
    gene_img = torch.squeeze(gene_img)
    k_img = fft_my(gene_img,(-2,-1))
    output_cat = torch.cat([torch.unsqueeze(k_img.real,1),torch.unsqueeze(k_img.imag,1)],dim=1)
    return output_cat

def ifft_my_2c(x_k):
    gene_k = torch.complex(x_k[:,0,:,:],x_k[:,1,:,:])
    gene_k = torch.squeeze(gene_k)
    img = fft_my(gene_k,(-2,-1))
    output_cat = torch.cat([torch.unsqueeze(img.real,1),torch.unsqueeze(img.imag,1)],dim=1)
    return output_cat

class DataConsistency_M(nn.Module):
    # DC layer
    def __init__(self):
        super(DataConsistency_M, self).__init__()

    def forward(self, x, x_k, mask):
        '''
        input [n 2 w h]
        x: predict image
        x_k: undersampled k sapce
        mask: original mask
        operation img space[n 2 w h ] - > [n w h ](complex type)
                  k space[n 2 w h ] - > [n w h ](complex type)
        output [n 2 w h]
        '''
        gene_img = torch.complex(x[:,0,:,:],x[:,1,:,:])
        gene_img = torch.squeeze(gene_img)
        # print("gene_img.shape",gene_img.shape,"mask.shape",mask.shape)
        gene_k = torch.complex(x_k[:,0,:,:],x_k[:,1,:,:])
        gene_k = torch.squeeze(gene_k)
        output = torch.rand(size=gene_img.size(), dtype=torch.complex64).cuda() #[n w h]
        # print("gene_img.size()",gene_img.size())
        if len(gene_img.size())==2: # just have [w h]
            gene_img = torch.unsqueeze(gene_img,0)
            gene_k = torch.unsqueeze(gene_k,0)
            output = torch.unsqueeze(output,0)
        for i in range(gene_img.size()[0]):
            pred_k = fft_my(gene_img[i],(-2,-1)) * (1.0 - mask[i])
            out_fft = gene_k[i] + pred_k
            output[i] = ifft_my(out_fft,(-2,-1)) #[n w h]
        output_cat = torch.cat([torch.unsqueeze(output.real,1),torch.unsqueeze(output.imag,1)],dim=1)
        # output_cat = torch.unsqueeze(output_cat,1)
        return output_cat

class DataConsistency_K(nn.Module):
    # DC layer
    def __init__(self):
        super(DataConsistency_K, self).__init__()

    def forward(self, x ,x_k, mask):
        '''
        input k [n 2 w h ] x_k [n 2 w h], mask [n w h]
        operation k space[n 2 w h ] - > [n w h ](complex type)
                  k space[n 2 w h ] - > [n w h ](complex type)
        output [n 2 w h]
        '''
        # x, x_k, mask  = input[0],input[1],input[2]
        gene_k = torch.complex(x[:,0,:,:],x[:,1,:,:])
        gene_k = torch.squeeze(gene_k) #[n w h]
        ori_k = torch.complex(x_k[:,0,:,:],x_k[:,1,:,:])
        ori_k = torch.squeeze(ori_k) #[n w h ]
        output = torch.rand(size=gene_k.size(),dtype=torch.complex64).cuda() #[n w h]
        # print("gene_img.size()",gene_img.size())
        for i in range(gene_k.size()[0]):
            pred_k = ori_k[i] * (1.0 - mask[i])
            out_fft = gene_k[i] + pred_k
            # print("out_fft.shape",out_fft.shape)
            output[i] = out_fft
        output_cat = torch.cat([torch.unsqueeze(output.real,1),torch.unsqueeze(output.imag,1)],dim=1)
        # output_cat = torch.unsqueeze(output_cat,1)
        return output_cat


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k=0, mask=0):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out_k = fft_my(logits,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [logits, out_k]


class UNetDC(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]
        # return [out]
        
        
class UNetDC_halfchannel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_halfchannel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(384, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(32, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]        

class UNetDC_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(128)
        self.awc3 = ECA_layer(64)        
        self.awc4 = ECA_layer(64)        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # 1024
        x = self.awc1(x) # 256
        x = self.up2(x, x3) # 512
        x = self.awc2(x)        
        x = self.up3(x, x2)
        x = self.awc3(x)
        x = self.up4(x, x1)
        x = self.awc4(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)        
        return [out,out_k]


class UNetDC_ECA_ASSP_1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_ECA_ASSP_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(128)
        self.awc3 = ECA_layer(64)        
        self.awc4 = ECA_layer(64)        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # 1024
        x = self.awc1(x) # 256
        x = self.up2(x, x3) # 512
        x = self.awc2(x)        
        x = self.up3(x, x2)
        x = self.awc3(x)
        x = self.up4(x, x1)
        x = self.awc4(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]


class UNetDC_ECA_ASSP_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_ECA_ASSP_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = ASPP_Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(128)
        self.awc3 = ECA_layer(64)        
        self.awc4 = ECA_layer(64)        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # 1024
        x = self.awc1(x) # 256
        x = self.up2(x, x3) # 512
        x = self.awc2(x)        
        x = self.up3(x, x2)
        x = self.awc3(x)
        x = self.up4(x, x1)
        x = self.awc4(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]
    
    
class UNetDC_ECA_ASSP_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_ECA_ASSP_3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = ASPP_Down(256, 512)
        self.down4 = Down(512, 512)
        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(128)
        self.awc3 = ECA_layer(64)        
        self.awc4 = ECA_layer(64)        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # 1024
        x = self.awc1(x) # 256
        x = self.up2(x, x3) # 512
        x = self.awc2(x)        
        x = self.up3(x, x2)
        x = self.awc3(x)
        x = self.up4(x, x1)
        x = self.awc4(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]    

class UNetDC_ECA_ASSP_4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_ECA_ASSP_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = ASPP_Down(512, 512)
        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(128)
        self.awc3 = ECA_layer(64)        
        self.awc4 = ECA_layer(64)        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc = DataConsistency_M()
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # 1024
        x = self.awc1(x) # 256
        x = self.up2(x, x3) # 512
        x = self.awc2(x)        
        x = self.up3(x, x2)
        x = self.awc3(x)
        x = self.up4(x, x1)
        x = self.awc4(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]

class UNetDC_short(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
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
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #64
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out, out_k]


class UNetDC_short_assp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_assp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Down(64, 128)
        self.down2 = ASPP_Down(128, 256)
        self.down3 = ASPP_Down(256, 512)
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
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #64
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]


class UNetDC_short_first_assp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_assp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Down(64, 128)
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
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #64
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)        
        return [out,out_k]

class UNetDC_short_mid_assp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_mid_assp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = ASPP_Down(128, 256)
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
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #64
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]

class UNetDC_short_last_assp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_last_assp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = ASPP_Down(256, 512)
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
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #64
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        return [out]


class UNetDC_short_first_assp_allECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_assp_allECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(128)
        self.awc2 = ECA_layer(256)
        self.awc3 = ECA_layer(512)

        self.awc4 = ECA_layer(256)
        self.awc5 = ECA_layer(64)
        self.awc6 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Down(64, 128)
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
        x2 = self.awc1(x2)        
        x3 = self.down2(x2) #256
        x3 = self.awc2(x3)
        x4 = self.down3(x3) #512
        x4 = self.awc3(x4)        
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3) #256
        x = self.awc4(x)
        x = self.up3(x, x2) #64 
        x = self.awc5(x)
        x = self.up4(x, x1) #64
        x = self.awc6(x)
        logits = self.outc(x)
        out = self.dc(logits,x_k,mask)
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]

class UNetDC_short_first_assp_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_first_assp_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Down(64, 128)
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


class UNetDC_short_mid_assp_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_mid_assp_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = ASPP_Down(128, 256)
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
        return [out]


class UNetDC_short_last_assp_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_last_assp_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = ASPP_Down(256, 512)
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
        return [out]



class UNetDC_short_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        
        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)
                
        self.down1 = Down(64, 128)
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

class UNetDC_short_ECA_assp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_short_ECA_assp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.awc1 = ECA_layer(256)
        self.awc2 = ECA_layer(64)
        self.awc3 = ECA_layer(64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ASPP_Down(64, 128)
        self.down2 = ASPP_Down(128, 256)
        self.down3 = ASPP_Down(256, 512)
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

class UNetDC_IK(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDC_IK, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc1 = DataConsistency_M()
        self.outc1 = OutConv(64, n_classes)
        self.outc2 = OutConv(64, n_classes)

        self.RAKI =  nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=[3, 5], padding=[1, 2], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], padding=[0, 0], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 3], padding=[0, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
        )

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc1(x)
        dc1_img = self.dc1(logits,x_k,mask) # img
        k_img1 = fft_my_2c(dc1_img) # k 
        k_img2 = k_img1
        for i in range(2):
            k_img2 = self.RAKI(k_img2) # k
        out = ifft_my_2c(k_img2) # img 
        dc2_img = self.dc1(out,x_k,mask) 
        # logits1 = self.outc1(x)

        return [dc2_img, dc1_img ]



class UNetunetDC_IK(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetunetDC_IK, self).__init__()
        self.name = "UNetunetDC_IK"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc1 = DataConsistency_M()
        self.outc1 = OutConv(64, n_classes)
        self.outc2 = OutConv(2, n_classes)

        self.unet_k =  UNet(n_channels, n_classes)

    def forward(self, x, x_k, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc1(x)
        dc1_img = self.dc1(logits,x_k,mask) # img
        k_img1 = fft_my_2c(dc1_img) # k 
        k_img2 = k_img1
        k_img2 = self.unet_k(k_img2,x_k,mask)[0] # k
        out = ifft_my_2c(k_img2) # img 
        dc2_img = self.dc1(out,x_k,mask) 
        dc2_img = self.outc2(dc2_img)

        return [dc2_img, dc1_img]