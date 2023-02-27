""" Full assembly of the parts to form the complete network """
from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from base_model.modules import IBN
from base_model.contrastive_head import ContrastiveHead_torch,  ContrastiveHead_myself

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_IBN(nn.Module):
    """
    Convolution Block IBN
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_IBN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            IBN(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            IBN(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

#Debug
class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.cat_(e4,d5)
        #d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat_(e3, d4)
        #d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat_(e2, d3)
        #d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat_(e1, d2)
        #d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)

        return d1

class UNet_IBN(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    UNet with IBN
    """

    def __init__(self, n_channels, n_classes):
        super(UNet_IBN, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_IBN(n_channels, filters[0])
        self.Conv2 = conv_block_IBN(filters[0], filters[1])
        self.Conv3 = conv_block_IBN(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.cat_(e4,d5)
        #d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat_(e3, d4)
        #d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat_(e2, d3)
        #d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat_(e1, d2)
        #d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)

        return d1

class UNet_DA(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes):
        super(UNet_DA, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.cat_(e4,d5)
        #d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat_(e3, d4)
        #d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat_(e2, d3)
        #d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat_(e1, d2)
        #d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)
        e5_logsoft = F.log_softmax(e5, dim=1)

        return d1, e5_logsoft

class UNet_contrast(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes):
        super(UNet_contrast, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()
        #self.contrast = ContrastiveHead_torch(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3) #init the contrast head to conv 8;
        self.contrast = ContrastiveHead_myself(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3)

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def forward(self, x, mask, trained,fake):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.cat_(e4,d5)
        #d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat_(e3, d4)
        #d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat_(e2, d3)
        #d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat_(e1, d2)
        #d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #d2 = d2 + contrast_tensor0
        out = self.Conv(d2)
        d1 = self.active(out)
        if trained and fake:
            contrast_tensor0, sample_sets, flag = self.contrast(d2,mask,trained,fake)
        elif trained and fake==False:
            contrast_tensor0, sample_sets, flag = self.contrast(d2, d1, trained, fake)
        else:
            contrast_tensor0, sample_sets, flag = self.contrast(d2,d1,trained,fake)

        return d1, sample_sets, flag


class UNet_contrastbase(nn.Module):
    """
    UNet - Basic Implementation Use region contrast
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes):
        super(UNet_contrastbase, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()
        #self.contrast = ContrastiveHead_torch(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3) #init the contrast head to conv 8;
        #self.contrast = ContrastiveHead_myself(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3)
        #self.contrast = ContrastiveHead_noedge(num_convs=1,num_projectfc=2,thred_u=0.2,scale_u=1.0,percent=0.3)
        self.contrast = ContrastiveHead_myself(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3)

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def forward(self, x, mask, trained,fake):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.cat_(e4,d5)
        #d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat_(e3, d4)
        #d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat_(e2, d3)
        #d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat_(e1, d2)
        #d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #d2 = d2 + contrast_tensor0
        out = self.Conv(d2)
        d1 = self.active(out)
        if trained and fake:
            contrast_tensor0, sample_sets, flag = self.contrast(d2,mask,trained,fake)
        elif trained and fake==False:
            contrast_tensor0, sample_sets, flag = self.contrast(d2, d1, trained, fake)
        else:
            contrast_tensor0, sample_sets, flag = self.contrast(d2,d1,trained,fake)

        return d1, sample_sets, flag, d2

