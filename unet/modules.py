import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(Conv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),            
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x) 

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, dilation=1, kernel_s = 3):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=kernel_s, stride=stride, padding=padding,
                dilation = dilation),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=dilation,dilation = dilation),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),            
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_s, stride=stride, padding=padding,
                dilation = dilation),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=21):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) #[n c 1 1]
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# Channel-wise attention
class CSE_Block(nn.Module):
    def __init__(self, in_channel, r, w, h):
        super(CSE_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.AvgPool2d((w, h)),
            nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.layer(x)
        return s*x

class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ASPP_Down(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[1, 2, 3],stride=[2,2,2,2]):
    # def __init__(self, in_dims, out_dims, rate=[1, 2, 3],stride=[1,1,1,1]):    
        super(ASPP_Down, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=stride[0], padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=stride[1], padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=stride[2], padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        # self.aspp_block4 = nn.Sequential(
        #     nn.Conv2d(
        #         in_dims, out_dims, 3, stride=stride[3], padding=rate[3], dilation=rate[3]
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(out_dims),
        # )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        # x4 = self.aspp_block4(x)
        # out = torch.cat([x1, x2, x3, x4], dim=1)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_Res(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[3, 5, 7]):
        super(ASPP_Res, self).__init__()

        self.aspp_block1 = ResidualConv(in_dims, out_dims, stride=1, padding=rate[0], dilation=rate[0])
        self.aspp_block2 = ResidualConv(in_dims, out_dims, stride=1, padding=rate[1], dilation=rate[1])
        self.aspp_block3 = ResidualConv(in_dims, out_dims, stride=1, padding=rate[2], dilation=rate[2])

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_Res_Down(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[1, 2, 3 ],stride=[2,2,2,2]):
        super(ASPP_Res_Down, self).__init__()

        self.aspp_block1 = ResidualConv(in_dims, out_dims, stride=stride[0], padding=rate[0], dilation=rate[0])
        self.aspp_block2 = ResidualConv(in_dims, out_dims, stride=stride[1], padding=rate[1], dilation=rate[1])
        self.aspp_block3 = ResidualConv(in_dims, out_dims, stride=stride[2], padding=rate[2], dilation=rate[2])
        # self.aspp_block4 = ResidualConv(in_dims, out_dims, stride=stride[3], padding=rate[3], dilation=rate[3])

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        # x4 = self.aspp_block4(x)        
        # out = torch.cat([x1, x2, x3, x4], dim=1)
        out = torch.cat([x1, x2, x3 ], dim=1)        
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class AttentionWeightChannel(nn.Module):

    def __init__(self, w, h, channel_num):
        super(AttentionWeightChannel, self).__init__()
        self.w = int(w)
        self.h = int(h)
        self.c = channel_num
        self.r = 16
        self.pool = nn.AvgPool2d((self.w, self.h))
        self.fc1 = nn.Linear(channel_num, channel_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_num // self.r, channel_num)
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(channel_num, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, inputs):
        x = torch.abs(inputs) # [n c w h]
        cs_x = self.cs_weight(inputs) # [n 1 w h]
        cs_x = self.sig(cs_x) # [n c w h]
        inputs_c = cs_x * inputs  #channel attention
         
        x = self.pool(x)    # [n c 1 1]    
        x = self.fc1(x.view(-1, x.shape[1])) # [n c ]

        weight = self.sig(x).view(-1,x.shape[1],1,1) # [n c ]
        inputs_s = weight * inputs #[n c ] * [n c w h ] = [n c w h] # spatial attention

        output = torch.max(inputs_c, inputs_s)  
        # return output, cs_x.detach(), weight.detach()
        return output



class AttentionWeightChannelB(nn.Module):

    def __init__(self, w, h, channel_num):
        super(AttentionWeightChannelB, self).__init__()
        self.w = int(w)
        self.h = int(h)
        self.c = channel_num
        self.r = 16

        self.pool = nn.AvgPool2d((self.w, self.h))
        self.fc1 = nn.Linear(channel_num, channel_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_num // self.r, channel_num)
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(channel_num, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, inputs):
        x = torch.abs(inputs) # [n c w h]
        cs_x = self.cs_weight(inputs) # [n 1 w h]
        cs_x = self.sig(cs_x) # [n c w h]
        inputs_c = cs_x * inputs  #channel attention
         
        x = self.pool(inputs_c)    # [n c 1 1]    
        x = self.fc1(x.view(-1, x.shape[1])) # [n c ]

        weight = self.sig(x).view(-1,x.shape[1],1,1) # [n c ]
        inputs_s = weight * inputs #[n c ] * [n c w h ] = [n c w h] # spatial attention

        output = inputs_s
        # return output, cs_x.detach(), weight.detach()
        return output


class AttentionWeightChannelC(nn.Module):

    def __init__(self, w, h, channel_num):
        super(AttentionWeightChannelB, self).__init__()
        self.w = int(w)
        self.h = int(h)
        self.c = channel_num
        self.r = 16

        self.pool = nn.AvgPool2d((self.w, self.h))
        self.fc1 = nn.Linear(channel_num, channel_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_num // self.r, channel_num)
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(channel_num, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, inputs):       
        x = self.pool(inputs)    # [n c 1 1]    
        x = self.fc1(x.view(-1, x.shape[1])) # [n c ]

        weight = self.sig(x).view(-1,x.shape[1],1,1) # [n c ]
        inputs_s = weight * inputs #[n c ] * [n c w h ] = [n c w h] # spatial attention

        x = torch.abs(inputs_s) # [n c w h]
        cs_x = self.cs_weight(inputs) # [n 1 w h]
        cs_x = self.sig(cs_x) # [n c w h]
        inputs_c = cs_x * inputs  #channel attention

        output = inputs_c
        # return output, cs_x.detach(), weight.detach()
        return output


class AttentionWeightChannelD(nn.Module):

    def __init__(self, w, h, channel_num):
        super(AttentionWeightChannelD, self).__init__()
        self.w = int(w)
        self.h = int(h)
        self.c = channel_num
        self.r = 16

        self.pool = nn.AvgPool2d((self.w, self.h))
        self.fc1 = nn.Linear(channel_num, channel_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_num // self.r, channel_num)
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(channel_num, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.eca = ECA_layer(channel_num)

    def forward(self, inputs):       
        x = self.pool(inputs)    # [n c 1 1]    
        x = self.fc1(x.view(-1, x.shape[1])) # [n c ]

        weight = self.sig(x).view(-1,x.shape[1],1,1) # [n c ]
        inputs_s = weight * inputs #[n c ] * [n c w h ] = [n c w h] # spatial attention

        x = torch.abs(inputs) # [n c w h]
        cs_x = self.cs_weight(inputs) # [n 1 w h]
        cs_x = self.sig(cs_x) # [n c w h]
        inputs_c = cs_x * inputs  #channel attention
        
        output = inputs_s*inputs_c*inputs
        # return output, cs_x.detach(), weight.detach()
        return output


class AttentionWeightChannelECA(nn.Module):

    def __init__(self, w, h, channel_num):
        super(AttentionWeightChannelECA, self).__init__()
        self.w = int(w)
        self.h = int(h)
        self.c = channel_num
        self.r = 16

        self.pool = nn.AvgPool2d((self.w, self.h))
        self.fc1 = nn.Linear(channel_num, channel_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_num // self.r, channel_num)
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(channel_num, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.eca = ECA_layer(channel_num)

    def forward(self, inputs):       
        x = self.pool(inputs)    # [n c 1 1]    
        x = self.fc1(x.view(-1, x.shape[1])) # [n c ]

        weight = self.sig(x).view(-1,x.shape[1],1,1) # [n c ]
        inputs_s = weight * inputs #[n c ] * [n c w h ] = [n c w h] # spatial attention

        x = torch.abs(inputs) # [n c w h]
        # cs_x = self.cs_weight(inputs) # [n 1 w h]
        # cs_x = self.sig(cs_x) # [n c w h]
        # inputs_c = cs_x * inputs  #channel attention
        inputs_c = self.eca(x)
        
        output = inputs_s*inputs_c*inputs
        # return output, cs_x.detach(), weight.detach()
        return output



class Res_Up_att(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=3, att_chan=24):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        # upsample = Upsample_(2)
        self.awc = AttentionWeightChannel(att_chan,att_chan, in_channels)
        self.conv =ResidualConv(in_channels, out_channels, stride = 1, padding=15, kernel_s=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.awc(x)
        return self.conv(x)
    
    
class Res_Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=3):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv =ResidualConv(in_channels, out_channels, stride = 1, padding=15, kernel_s=kernel_size)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)