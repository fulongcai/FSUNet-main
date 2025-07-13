import torch
import torch.nn as nn
import torch.nn.functional as F
from .DCNv2.dcn_v2 import DCN
from .dilated_convs import dilated_conv, dilated_conv_d


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class Freq_fft(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(Freq_fft, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels

        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = x + 0.000000000001

        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        Amp = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        Amp = self.conv_1(Amp)
        Amp = self.leaky(Amp)
        Amp = self.conv_2(Amp)
        Amp = torch.tanh(Amp)
        real = Amp * torch.cos(pha)
        imag = Amp * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_freq = out + x
        return x_freq

class dilated_conv_2d(nn.Module):
    def __init__(self, filters, kernel_size, padding, dilation):
        super(dilated_conv_2d, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

        self.dilated_conv_d = dilated_conv_d(channels=self.filters, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation)
        self.Batch = nn.BatchNorm2d(self.filters)
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        output = self.ReLU(self.Batch(self.dilated_conv_d(x)))
        return output

class H_GC(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(H_GC, self).__init__()
        self.in_channels = in_channels // 2
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )

        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )

        self.dcn = DCN(in_channels // 2, in_channels // 2, kernel_size, stride, padding)

        self.bn = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

        self.point_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.conv_channel = nn.Conv2d(in_channels // 2, 1, 1, 1)

    def apply_sobel(self, x):
        sobel_kernel_x = self.sobel_kernel_x.to(x.device).repeat(x.size(1), 1, 1, 1)
        sobel_kernel_y = self.sobel_kernel_y.to(x.device).repeat(x.size(1), 1, 1, 1)

        grad_x = F.conv2d(x, sobel_kernel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_kernel_y, padding=1, groups=x.size(1))

        grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-8)
        grad_y = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min() + 1e-8)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return edge_map

    def forward(self, x):
        x_high = x[:, self.in_channels:, :, :]

        hat_x_edge = self.apply_sobel(x_high)

        hat_x_edge = F.pad(hat_x_edge, (0, 1, 0, 1))
        significant_x_edge = self.max_pool(hat_x_edge) - self.avg_pool(hat_x_edge)

        significant_x_edge = significant_x_edge.contiguous()
        dcn_out = self.dcn(x_high)

        dcn_out = self.bn(dcn_out)
        dcn_out = self.relu(dcn_out)

        f_high = torch.cat([dcn_out, significant_x_edge], dim=1)
        f_high = self.point_conv(f_high)
        f_high = self.conv_channel(f_high).sigmoid()
        return f_high


class L_SR(nn.Module):
    def __init__(self, in_channels):
        super(L_SR, self).__init__()
        self.dilated_conv0 = dilated_conv_2d(in_channels, 3, 3, 3)
        self.dilated_conv1 = dilated_conv_2d(in_channels, 3, 5, 5)
        self.dilated_conv2 = dilated_conv_2d(in_channels, 3, 7, 7)
        self.dilated_conv3 = nn.Sequential(
            dilated_conv(in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.ConV = DoubleConv(in_channels, in_channels, in_channels // 2)
        self.conv_channel = nn.Conv2d(in_channels, 1, 1, 1)
        self.fft_to_spatial = Freq_fft(in_channels)
    def forward(self, x_low):
        x_3 = self.dilated_conv3(x_low)

        x_0 = self.dilated_conv0(x_low)
        x_1 = self.dilated_conv1(x_low)
        x_2 = self.dilated_conv2(x_low)

        x_all = x_0 + x_1 + x_2 + x_3
        x_multi = self.ConV(x_all)
        channle_att = self.conv_channel(x_multi).sigmoid() + 0.0001
        x_multi = x_multi * channle_att
        # x_freq = self.fft_to_spatial(x_low)
        f_low = x_multi

        return f_low

