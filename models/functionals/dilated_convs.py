import torch.nn as nn
import torch.nn.functional as F


class Avg_ChannelAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # bz,C_out,h,w -> bz,C_out,1,1
            nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1, 1),
                nn.Conv2d(channels * 2, channels * 2 // r, 1, 1, 0, bias=False),  # bz,C_out,1,1 -> bz,C_out/r,1,1
                nn.Conv2d(channels * 2 // r,channels // r,1,1)
            ),
            # nn.Conv2d(channels, channels // r,1,1,0,bias=False),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0, bias=False),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1, 1),
                nn.BatchNorm2d(channels * 2),
                nn.Conv2d(channels * 2, channels, 1, 1),
            ),
            # nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.avg_channel(x)


class dilated_conv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(dilated_conv, self).__init__()

        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        out_normal = self.conv(x)  # 完整的卷积结果
        theta = self.attn(x)

        kernel_w1 = self.conv.weight.sum(2).sum(2)
        kernel_w2 = kernel_w1[:, :, None, None]

        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)  # 全局加权卷积，卷积核的所有权重聚合为全局加权值，提升特征的全局贡献
        # Filter the feature with $\textbf{W}_{c}$
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)  # 卷积核的中心点权重，提取特征的局部响应

        return out_center - out_normal + theta * out_offset


class dilated_conv_d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=False):
        super(dilated_conv_d, self).__init__()

        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        out_normal = self.conv(x)
        theta = self.attn(x)

        conv_layer = self.conv
        kernel_w1 = conv_layer.weight.sum(2).sum(2)
        kernel_w2 = kernel_w1[:, :, None, None]
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=conv_layer.bias, stride=conv_layer.stride,
                              padding=0, groups=conv_layer.groups)
        center_w1 = conv_layer.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=conv_layer.bias, stride=conv_layer.stride,
                              padding=0, groups=conv_layer.groups)
        return out_center - out_normal + theta * out_offset
