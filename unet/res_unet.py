import torch
import torch.nn as nn
from .modules import ResidualConv, Upsample
from .unet_model import DataConsistency_M, fft_my_2c, ifft_my_2c

class ResUnet(nn.Module):
    def __init__(self, channel, out_channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.name = "ResUnet"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_k, mask):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
 
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return [output]




class ResUnet_DC(nn.Module):
    def __init__(self, channel, out_channel, filters=[64, 128, 256, 512]):
        super(ResUnet_DC, self).__init__()
        self.name = "ResUnet_DC"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            # nn.Sigmoid(),
        )
        self.dc = DataConsistency_M()


    def forward(self, x, x_k, mask):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge     Decode
        x4 = self.bridge(x3)
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
 
        x6 = self.up_residual_conv1(x5)
        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)
        x8 = self.upsample_3(x8)
        
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)
        output = self.dc(output,x_k,mask)
        return [output]



class ResUnet_DC_Normal(nn.Module):
    def __init__(self, channel, out_channel, filters=[64, 128, 256, 512, 1024]):
        super(ResUnet_DC_Normal, self).__init__()
        self.name = "ResUnet_DC_Normal"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4= Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)        

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            # nn.Sigmoid(),
        )
        self.dc = DataConsistency_M()


    def forward(self, x, x_k, mask):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)        
        # Bridge
        x5 = self.bridge(x4)
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)
 
        x7 = self.up_residual_conv1(x6)
        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)
        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)
        
        x11 = self.up_residual_conv3(x10)
        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)
        
        x13 = self.up_residual_conv4(x12)
        output = self.output_layer(x13)
        output = self.dc(output,x_k,mask)
        return [output]



class ResUnet_DCIK(nn.Module):
    def __init__(self, channel, out_channel, filters=[64, 128, 256, 512]):
        super(ResUnet_DCIK, self).__init__()
        self.name = "ResUnet_DCIK"

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            # nn.Sigmoid(),
        )
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


    def forward(self, x, x_k, mask):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
 
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)
        dc1_img = self.dc(output,x_k,mask)

        dc1_k = ifft_my_2c(dc1_img) # img 
        k_img2 = dc1_k
        output = self.RAKI(k_img2)
        dc2_img = fft_my_2c(output)
        dc2_img = self.dc(dc2_img,x_k,mask)

        return [dc2_img,dc1_img]