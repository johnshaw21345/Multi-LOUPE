import torch.nn as nn
import torch
import numpy as np
from .modules import (
    ResidualConv,
    Conv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)
from .unet_parts import *
from .unet_model import DataConsistency_M, DataConsistency_K
from utils.utils import fft_my, ifft_my

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    :param k: input in k-space
    :param k0: initially sampled elements in k-space
    :param mask: corresponding non-zero location
    :param noise_lvl:
    :return:
    """

    if noise_lvl:
        out = (1 - mask) * k + mask * (k + noise_lvl * k0) / (1 + noise_lvl)
    else:
        out = (1 - mask) * k + mask * k0

    return out

class DataConsistencyInKspace(nn.Module):
    """
    Create data consistency operator
    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()

        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        :param x: input in image domain, of shape (n, 2, nx, ny, nt)
        :param k0: initially sampled elements in k-space
        :param mask: corresponding nonzero location
        :return:
        """
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)

        elif x.dim() == 5:
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = torch.fft.fft(x, 2, normalized=self.normalized)
        #k = torch.fft.fft2(x, )
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.fft.ifft(out, 2, normalized=self.normalized)
        #x_res = torch.fft.ifft2(out, 2, normalized=self.normalized)
        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

# conv blocks for CNNs
def conv_block(n_ch,                # input channel
                n_dims,                 # layers of CNNs
                n_feats=32,            # feature layers
                kernel_size=3,         # CNN kernel size
                dilation=1,            # dilated convolution
                bn=False,               # batchnormalization
                nl='lrelu',            # activation function
                conv_dim=2,             # 2DConv or 3DConv
                n_out=None):           # output channel

    # decide to use 2D or 3D
    if conv_dim == 2:
        conv = nn.Conv2d
    elif conv_dim == 3:
        conv = nn.Conv3d
    else:
        print("Wrong conv dimension.")

    # define output channel
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    # define activation function
    af = relu if nl == "relu" else lrelu

    # define CNNs in the middle
    def conv_i():
        return conv(n_feats, n_feats, kernel_size, stride=1,
                    padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, n_feats, kernel_size, stride=1,
                  padding=pad_conv, bias=True)
    conv_n = conv(n_feats, n_out, kernel_size, stride=1,
                  padding=pad_conv, bias=True)

    # fill in the CNNs
    layers = [conv_1, af()]

    for i in range(n_dims - 2):
        layers += [conv_i(), af(), nn.BatchNorm2d(n_feats)]

    layers += [conv_n]
    # the whole squence is [conv relu  conv relu bn * n conv]
    return nn.Sequential(*layers)


class DCCNN(nn.Module):
    def __init__(self,
                 nc_dims=5,             # the depth of cascade
                 nd_dims=5,             # the number of convolution layers
                 n_channels=2):
        super(DCCNN, self).__init__()

        self.nc = nc_dims
        self.nd = nd_dims

        print('A cascade network with {} CNNs and {} DCs'.format(nc_dims, nd_dims))

        blocks = []
        dcs = []

        for i in range(nc_dims):
            # add the numbers of convolution layers
            blocks.append(conv_block(n_channels, nd_dims))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.blocks = nn.ModuleList(blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.blocks[i](x)
            x += x_cnn
            x = self.dcs[i].perform(x, k, m)
        return x


class CascadeMRI(nn.Module):
    def __init__(self, in_channels, out_channels,
                nc_dims=5,             # the depth of cascade
                nd_dims=5,             # the number of convolution layers
                DC=True):
        super(CascadeMRI, self).__init__()
        self.nc = nc_dims
        self.nd = nd_dims
        self.ouc = OutConv(32, out_channels)
        self.DC = DC

        def gen_blocks(nd_dims):
            cnn_list = [ResidualConv(in_channels,48,stride=1, padding=1)]
            for i in range(nd_dims - 2):
                cnn_list.append(ResidualConv(48,48,stride=1, padding=1))
            cnn_list.append(ResidualConv(48,out_channels,stride=1, padding=1))
            return nn.Sequential(*cnn_list)
        
        blocks = []
        dcs = []

        for i in range(nc_dims):
            # add the numbers of convolution layers
            blocks.append(gen_blocks(nd_dims))
            dcs.append(DataConsistency_M())

        self.blocks = nn.ModuleList(blocks)
        self.dcs = dcs

    def forward(self, x, X_k, mask):
        for i in range(self.nc):
            x_cnn = self.blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i](x, X_k, mask)
        return [x]


class CascadeMRIM(nn.Module):
    def __init__(self, in_channels, out_channels,
                nc_dims=5,             # the depth of cascade
                nd_dims=5,             # the number of convolution layers
                DC=True):
        super(CascadeMRIM, self).__init__()
        self.nc = nc_dims
        self.nd = nd_dims
        self.ouc = OutConv(32, out_channels)
        self.DC = DC

        def gen_blocks(nd_dims):
            cnn_list = [ResidualConv(in_channels,48,stride=1, padding=1)]
            for i in range(nd_dims - 2):
                cnn_list.append(ResidualConv(48,48,stride=1, padding=1))
            cnn_list.append(ResidualConv(48,out_channels,stride=1, padding=1))
            return nn.Sequential(*cnn_list)
        
        blocks = []
        self.dcs = []

        for i in range(nc_dims):
            # add the numbers of convolution layers
            blocks.append(gen_blocks(nd_dims))
            self.dcs.append(DataConsistency_M())
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, X_k, mask):
        temp = x
        for index in range(len(self.dcs)):
            conv5 = self.blocks[index](temp)
            # block = temp + conv5
            # print("block",conv5.size())
            temp = self.dcs[index](conv5,X_k,mask)
        # out = self.ouc(block)
        out = temp
        out_k = fft_my(out,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)
        return [out,out_k]


class CascadeMRI_K(nn.Module):
    def __init__(self, in_channels, out_channels,DC=False):
        super(CascadeMRI_K, self).__init__()
        self.ouc = OutConv(32, out_channels)
        self.DC = DC
        cmplx_2d1 = nn.Sequential( 
            # Conv(in_channels,48),
            ResidualConv(in_channels,48,stride=1, padding=1),
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,out_channels,stride=1, padding=1),  
        )

        cmplx_2d2 = nn.Sequential( 
            ResidualConv(in_channels,48,stride=1, padding=1),
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,out_channels,stride=1, padding=1),   
        )

        cmplx_2d3 = nn.Sequential( 
            ResidualConv(in_channels,48,stride=1, padding=1),
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,out_channels,stride=1, padding=1),  
        )

        cmplx_2d4 = nn.Sequential( 
            ResidualConv(in_channels,48,stride=1, padding=1),
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,out_channels,stride=1, padding=1),  
        )

        cmplx_2d5 = nn.Sequential( 
            ResidualConv(in_channels,48,stride=1, padding=1),
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,48,stride=1, padding=1),  
            ResidualConv(48,out_channels,stride=1, padding=1),  
        )

        self.op_list = nn.ModuleList([cmplx_2d1,cmplx_2d2,cmplx_2d3,cmplx_2d4,cmplx_2d5])
        self.dc = DataConsistency_K()

    def forward(self, x, X_k, mask):
        temp = x
        for index in range(len(self.op_list)):
            conv5 = self.op_list[index](temp)
            block = temp + conv5
            print("block",conv5.size())
            temp = self.dc(block,X_k,mask)
        # out = self.ouc(block)
        out = temp
        return [out]


if __name__ == '__main__':
    net = CascadeMRI(2,2).cuda()
    inp = torch.randn((2,2,96,96)).cuda()
    mask = torch.randn((96,96)).cuda()
    x = net.forward(inp,inp,mask)