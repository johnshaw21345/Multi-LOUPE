import torch.nn as nn
import torch
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
    ECA_layer,
    Res_Up,
    Res_Up_att
)
from .unet_model import DataConsistency_M, fft_my_2c, ifft_my_2c, DataConsistency_K, UNet
from .cascade_net import CascadeMRI, CascadeMRI_K
from utils.utils import fft_my, ifft_my

class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()
        self.name = "ResUnetPlusPlus"
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        return [out]



class ResUnetPlusPlus_DC(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DC, self).__init__()
        self.name = "ResUnetPlusPlus_DC"
        print("input channel is ",channel)
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

        # self.RAKI =  nn.Sequential(
        #     nn.BatchNorm2d(2),
        #     nn.Conv2d(in_channels=2, out_channels=128, kernel_size=[3, 5], padding=[1, 2], padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], padding=[0, 0], padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 3], padding=[0, 1], padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
        # )

        # self.RAKI =  nn.Sequential(
        #     ResidualConv(out_channel,64,1,1),
        #     ResidualConv(64,128,1,1),
        #     ResidualConv(128,256,1,1),
        #     ResidualConv(256,512,1,1),
        #     ResidualConv(512,256,1,1),
        #     ResidualConv(256,128,1,1),
        #     ResidualConv(128,64,1,1),
        #     ResidualConv(64,32,1,1),
        #     ResidualConv(32,out_channel,1,1),
        # )
        # self.RAKI = CascadeMRI(2,2)


    def forward(self, x, x_k, mask):
        # print("self.input_skip(x) ",self.input_skip(x).size())
        # print("self.input_layer(x) ",self.input_layer(x).size())
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        # dc1_k = ifft_my_2c(dc1_img) # img 
        # k_img2 = dc1_k
        # output = self.RAKI(k_img2)
        # dc2_img = fft_my_2c(output)
        # dc2_img = self.dc(dc2_img,x_k,mask)
        
        return [dc1_img]

class ResUnetPlusPlus_DCIK(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DCIK, self).__init__()
        self.name = "ResUnetPlusPlus_DCIK"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

        # self.RAKI =  nn.Sequential(
        #     nn.BatchNorm2d(2),
        #     nn.Conv2d(in_channels=2, out_channels=128, kernel_size=[3, 5], padding=[1, 2], padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], padding=[0, 0], padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 3], padding=[0, 1], padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
        # )
        # self.RAKI = UNet(2,2)
        self.RAKI =  nn.Sequential(
            ResidualConv(out_channel,64,1,1),
            ResidualConv(64,128,1,1),
            ResidualConv(128,256,1,1),
            ResidualConv(256,512,1,1),
            ResidualConv(512,256,1,1),
            ResidualConv(256,128,1,1),
            ResidualConv(128,64,1,1),
            ResidualConv(64,32,1,1),
            ResidualConv(32,out_channel,1,1),
        )  
        # self.RAKI = CascadeMRI_K(channel,out_channel)
        # self.attion = AttentionWeightChannel(96,96,4)
    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        dc1_k = ifft_my_2c(dc1_img) # img 
        k_img2 = dc1_k
        output = self.RAKI(k_img2)

        # output = self.RAKI(k_img2,x_k,mask)[0]
        # print("output.shape",output.shape)
        dc2_img = fft_my_2c(output)
        dc2_img = self.dc(dc2_img,x_k,mask)
        # out = torch.concat([dc2_img,dc1_img],dim=1)
        # out = self.attion(out)
        # out = self.dc(out,x_k,mask)
        return [dc2_img,dc1_img]


class ResUnetPlusPlus_DC_att(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DC_att, self).__init__()
        self.name = "ResUnetPlusPlus_DC_att"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.upsample1 = Upsample_(2)
        self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.upsample2 = Upsample_(2)
        self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.upsample3 = Upsample_(2)
        self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x2) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x3) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x4) # 256 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]



class ResUnetPlusPlus_DC_att_Bigkernel(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DC_att_Bigkernel, self).__init__()
        self.name = "ResUnetPlusPlus_DC_att_Bigkernel"
        self.kernel_size = 31
        self.padding = [15,15,11,11,11]
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=self.kernel_size, padding=self.padding[0]),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=self.kernel_size, padding=self.padding[1]),)
        self.input_skip = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=self.kernel_size, padding=self.padding[0]))

        self.final_skip = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=self.kernel_size, padding=self.padding[0]))

        self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], stride = 2, padding=15,kernel_s=self.kernel_size)
        self.residual_conv2 = ResidualConv(filters[1], filters[2], stride = 2, padding=15,kernel_s=self.kernel_size)
        self.residual_conv3 = ResidualConv(filters[2], filters[3], stride = 2, padding=15,kernel_s=self.kernel_size)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3]) 1 
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.up1 = Res_Up_att(filters[4] + filters[2],filters[3], kernel_size=self.kernel_size, att_chan=24)
        # self.upsample1 = Upsample_(2)
        # self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        # self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], stride = 1, padding=15, kernel_s=self.kernel_size)

        self.up2 = Res_Up_att(filters[3] + filters[1], filters[2], kernel_size=self.kernel_size, att_chan=48)
        # self.upsample2 = Upsample_(2)
        # self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        # self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], stride = 1, padding=15, kernel_s=self.kernel_size)

        self.up3 = Res_Up_att(filters[2] + filters[0],filters[1], kernel_size=self.kernel_size, att_chan=96)
        # self.upsample3 = Upsample_(2)
        # self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])
        # self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], stride = 1, padding=15, kernel_s=self.kernel_size)

        self.aspp_out = ASPP(filters[1], filters[0])
        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        # x1 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x1) # 64 48 48
        # print("x2.shape",x2.shape)
        # x3 = self.squeeze_excite2(x2)
        # x2 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x2) # 128 24 24
        # print("x3.shape",x3.shape)
        # x4 = self.squeeze_excite3(x3)
        # x3 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x3) # 256 12 12
        # print("x4.shape",x4.shape)
        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        x6 = self.up1(x5, x3)
        # x6 = self.upsample1(x5)
        # x6 = torch.cat([x6, x3], dim=1)
        # x6 = self.awc1(x6)
        # x6 = self.up_residual_conv1(x6)

        x7 = self.up2(x6, x2)
        # x7 = self.upsample2(x6)
        # x7 = torch.cat([x7, x2], dim=1)
        # x7 = self.awc2(x7)
        # x7 = self.up_residual_conv2(x7)

        x8 = self.up3(x7, x1)
        # x8 = self.upsample3(x7)
        # x8 = torch.cat([x8, x1], dim=1)
        # x8 = self.awc3(x8)
        # x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)
        out_k = fft_my(dc1_img,(-2,-1))
        out_k = torch.stack([out_k.real,out_k.imag],axis=1)

        return [dc1_img, out_k]


class ResUnetPlusPlus_DC_att_Medkernel(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DC_att_Bigkernel, self).__init__()
        self.name = "ResUnetPlusPlus_DC_att_Medkernel"
        self.kernel_size = 31
        self.padding = [15,15,11,11,11]
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=self.kernel_size, padding=self.padding[0]),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=self.kernel_size, padding=self.padding[1]),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=self.kernel_size, padding=self.padding[0])
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=self.kernel_size, padding=self.padding[0])
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], stride = 2, padding=15,kernel_s=self.kernel_size)
        self.residual_conv2 = ResidualConv(filters[1], filters[2], stride = 2, padding=15,kernel_s=self.kernel_size)
        self.residual_conv3 = ResidualConv(filters[2], filters[3], stride = 2, padding=15,kernel_s=self.kernel_size)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.upsample1 = Upsample_(2)
        self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], stride = 1, padding=15, kernel_s=self.kernel_size)

        self.upsample2 = Upsample_(2)
        self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], stride = 1, padding=15, kernel_s=self.kernel_size)

        self.upsample3 = Upsample_(2)
        self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], stride = 1, padding=15, kernel_s=self.kernel_size)

        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x2) # 64 48 48
        print("x2.shape",x2.shape)
        # x3 = self.squeeze_excite2(x2)
        x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x3) # 128 24 24
        print("x3.shape",x3.shape)
        # x4 = self.squeeze_excite3(x3)
        x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x4) # 256 12 12
        print("x4.shape",x4.shape)
        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]

class ResUnetPlusPlus_DC_att_assp(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DC_att_assp, self).__init__()
        self.name = "ResUnetPlusPlus_DC_att_assp"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ASPP_Res_Down(filters[0], filters[1])
        self.residual_conv2 = ASPP_Res_Down(filters[1], filters[2])
        self.residual_conv3 = ASPP_Res_Down(filters[2], filters[3])

        self.aspp_bridge = ASPP_Res(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.awc1 = ECA_layer(filters[4] + filters[2])
        self.awc2 = ECA_layer(filters[3] + filters[1])
        self.awc3 = ECA_layer(filters[2] + filters[0])

        # self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        # self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        # self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])

        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ASPP_Res(filters[4] + filters[2], filters[3])

        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ASPP_Res(filters[3] + filters[1], filters[2])

        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ASPP_Res(filters[2] + filters[0], filters[1])

        self.aspp_out = ASPP_Res(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))
        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        # x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x1) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        # x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x2) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        # x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x3) # 256 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]


class ResUnetPlusPlus_DC_att_assp_normal(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512, 1024]):
        super(ResUnetPlusPlus_DC_att_assp_normal, self).__init__()
        self.name = "ResUnetPlusPlus_DC_att_assp_normal"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ASPP_Res_Down(filters[0], filters[1])
        self.residual_conv2 = ASPP_Res_Down(filters[1], filters[2])
        self.residual_conv3 = ASPP_Res_Down(filters[2], filters[3])
        self.residual_conv4 = ASPP_Res_Down(filters[3], filters[4])

        self.aspp_bridge = ASPP_Res(filters[4], filters[5])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.awc0 = ECA_layer(filters[5] + filters[3])
        self.awc1 = ECA_layer(filters[4] + filters[2])
        self.awc2 = ECA_layer(filters[3] + filters[1])
        self.awc3 = ECA_layer(filters[2] + filters[0])

        # self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        # self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        # self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])

        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ASPP_Res(filters[5] + filters[3], filters[4])

        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ASPP_Res(filters[4] + filters[2], filters[3])

        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ASPP_Res(filters[3] + filters[1], filters[2])

        self.upsample4 = Upsample_(2)
        self.up_residual_conv4 = ASPP_Res(filters[2] + filters[0], filters[1])
        
        self.aspp_out = ASPP_Res(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))
        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        # x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x1) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        # x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x2) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        # x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x3) # 256 12 12

        x5 = self.residual_conv4(x4) # 512 6 6

        x6 = self.aspp_bridge(x5) # 1024 6 6

        # x6 = self.attn1(x3, x5)
        x7 = self.upsample1(x6) # 1024 12 12
        x7 = torch.cat([x7, x4], dim=1) # 1024+512 12
        x7 = self.awc0(x7)
        x7 = self.up_residual_conv1(x7)  # 512 12

        # x7 = self.attn2(x2, x6)
        x8 = self.upsample2(x7) # 512 24
        x8 = torch.cat([x8, x3], dim=1) #512+128 24
        x8 = self.awc1(x8)
        x8 = self.up_residual_conv2(x8) #256 24

        # x8 = self.attn3(x1, x7)
        x9 = self.upsample3(x8) # 256 48
        x9 = torch.cat([x9, x2], dim=1)
        x9 = self.awc2(x9)
        x9 = self.up_residual_conv3(x9) # 128 48

        x10 = self.upsample3(x9) # 128 96
        x10 = torch.cat([x10, x1], dim=1)
        x10 = self.awc3(x10) #128+32 96
        x10 = self.up_residual_conv4(x10) #64 96

        x11 = self.aspp_out(x10)
        out = self.output_layer(x11)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]

class ResUnetPlusPlus_DC_att_asspsame(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DC_att_asspsame, self).__init__()
        self.name = "ResUnetPlusPlus_DC_att_asspsame"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ASPP_Res(filters[0], filters[1])
        self.residual_conv2 = ASPP_Res(filters[1], filters[2])
        self.residual_conv3 = ASPP_Res(filters[2], filters[3])

        self.aspp_bridge = ASPP_Res(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.awc1 = ECA_layer(filters[4] + filters[2])
        self.awc2 = ECA_layer(filters[3] + filters[1])
        self.awc3 = ECA_layer(filters[2] + filters[0])

        # self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        # self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        # self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])

        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ASPP_Res(filters[4] + filters[2], filters[3])

        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ASPP_Res(filters[3] + filters[1], filters[2])

        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ASPP_Res(filters[2] + filters[0], filters[1])

        self.aspp_out = ASPP_Res(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))
        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        # x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x1) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        # x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x2) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        # x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x3) # 256 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        # x6 = self.upsample1(x5)
        x6 = x5
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        # x7 = self.upsample2(x6)
        x7 = x6
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        # x8 = self.upsample3(x7)
        x8 = x7
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]


class CAU_NetV0(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(CAU_NetV0, self).__init__()
        self.name = "CAU_NetV0"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.awc1_in = AttentionWeightChannel(48,48,filters[1])
        self.awc2_in = AttentionWeightChannel(24,24,filters[2])
        self.awc3_in = AttentionWeightChannel(12,12,filters[3])

        self.residual_conv1 = ASPP_Res_Down(filters[0], filters[1])
        self.residual_conv2 = ASPP_Res_Down(filters[1], filters[2])
        self.residual_conv3 = ASPP_Res_Down(filters[2], filters[3])

        self.aspp_bridge = ASPP_Res(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.upsample1 = Upsample_(2)
        self.awc1 = AttentionWeightChannel(24,24,filters[3])
        self.up_residual_conv1 = ASPP_Res(filters[4] + filters[2], filters[3])

        self.upsample2 = Upsample_(2)
        self.awc2 = AttentionWeightChannel(48,48,filters[2])
        self.up_residual_conv2 = ASPP_Res(filters[3] + filters[1], filters[2])

        self.upsample3 = Upsample_(2)
        self.awc3 = AttentionWeightChannel(96,96,filters[1])
        self.up_residual_conv3 = ASPP_Res(filters[2] + filters[0], filters[1])

        self.aspp_out = ASPP_Res(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)

        x2 = self.residual_conv1(x1) # 64 48 48
        x2 = self.awc1_in(x2) # 32 48 48

        # x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x2) # 128 24 24
        x3 = self.awc2_in(x3) # 64 24 24

        # x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x3) # 256 12 12
        x4 = self.awc3_in(x4) # 128 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6) # 64 24 24
        x6 = self.awc1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)
        x7 = self.awc2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)
        x8 = self.awc3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]


class CAU_NetV1(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(CAU_NetV1, self).__init__()
        self.name = "CAU_NetV1"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = Conv(filters[0], filters[1], 2, 1)
        self.residual_conv2 = Conv(filters[1], filters[2], 2, 1)
        self.residual_conv3 = Conv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.upsample1 = Upsample_(2)
        self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        self.up_residual_conv1 = Conv(filters[4] + filters[2], filters[3],1,1)

        self.upsample2 = Upsample_(2)
        self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        self.up_residual_conv2 = Conv(filters[3] + filters[1], filters[2],1,1)

        self.upsample3 = Upsample_(2)
        self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])
        self.up_residual_conv3 = Conv(filters[2] + filters[0], filters[1],1,1)
        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x2) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x3) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x4) # 256 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]



class CAU_NetV2(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(CAU_NetV2, self).__init__()
        self.name = "CAU_NetV2"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ASPP_Down(filters[0], filters[1])
        self.residual_conv2 = ASPP_Down(filters[1], filters[2])
        self.residual_conv3 = ASPP_Down(filters[2], filters[3])

        self.aspp_bridge = ASPP(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.upsample1 = Upsample_(2)
        # self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        self.up_residual_conv1 = ASPP(filters[4] + filters[2], filters[3])

        self.upsample2 = Upsample_(2)
        # self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        self.up_residual_conv2 = ASPP(filters[3] + filters[1], filters[2])

        self.upsample3 = Upsample_(2)
        # self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])
        self.up_residual_conv3 = ASPP(filters[2] + filters[0], filters[1])
        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        # x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x1) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        # x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x2) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        # x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x3) # 256 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        # x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        # x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        # x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]


class CAU_NetV3(nn.Module):
    def __init__(self, channel, out_channel, filters=[32, 64, 128, 256, 512]):
        super(CAU_NetV3, self).__init__()
        self.name = "CAU_NetV3"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.final_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        # self.awc1_in = AttentionWeightChannel(96,96,filters[0])
        # self.awc2_in = AttentionWeightChannel(48,48,filters[1])
        # self.awc3_in = AttentionWeightChannel(24,24,filters[2])

        self.residual_conv1 = ASPP_Res_Down(filters[0], filters[1])
        self.residual_conv2 = ASPP_Res_Down(filters[1], filters[2])
        self.residual_conv3 = ASPP_Res_Down(filters[2], filters[3])

        self.aspp_bridge = ASPP_Res(filters[3], filters[4])

        # self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        # self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        # self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])

        self.upsample1 = Upsample_(2)
        # self.awc1 = AttentionWeightChannel(24,24,filters[4] + filters[2])
        self.up_residual_conv1 = ASPP_Res(filters[4] + filters[2], filters[3])

        self.upsample2 = Upsample_(2)
        # self.awc2 = AttentionWeightChannel(48,48,filters[3] + filters[1])
        self.up_residual_conv2 = ASPP_Res(filters[3] + filters[1], filters[2])

        self.upsample3 = Upsample_(2)
        # self.awc3 = AttentionWeightChannel(96,96,filters[2] + filters[0])
        self.up_residual_conv3 = ASPP_Res(filters[2] + filters[0], filters[1])
        self.aspp_out = ASPP_Res(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], out_channel, 1))

        self.dc = DataConsistency_M()

    def forward(self, x, x_k, mask):
        x1 = self.input_layer(x) + self.input_skip(x) # 32 96 96

        # x2 = self.squeeze_excite1(x1)
        # x2 = self.awc1_in(x1) # 32 96 96
        x2 = self.residual_conv1(x1) # 64 48 48

        # x3 = self.squeeze_excite2(x2)
        # x3 = self.awc2_in(x2) # 64 48 48
        x3 = self.residual_conv2(x2) # 128 24 24

        # x4 = self.squeeze_excite3(x3)
        # x4 = self.awc3_in(x3) # 128 24 24
        x4 = self.residual_conv3(x3) # 256 12 12

        # x5 = self.aspp_bridge(x4) # 512 12 12
        x5 = self.aspp_bridge(x4) # 512 12 12

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        # x6 = self.awc1(x6)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        # x7 = self.awc2(x7)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        # x8 = self.awc3(x8)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        dc1_img = self.dc(out,x_k,mask)

        return [dc1_img]