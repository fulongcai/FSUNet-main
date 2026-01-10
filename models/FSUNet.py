import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch.nn import BCELoss

from .functionals.DCNv2.dcn_v2 import DCN
from .functionals.encoder import encoder
from .functionals.focalloss import FocalLoss, EdgeLoss
from .functionals import GatedSpatialConv as gsc
from .functionals.DBR import H_GC, L_SR

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

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.conv_1 = DCN(in_channels=in_channels,  out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.conv_2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x2 = x2.contiguous()
        x2 = self.conv_2(self.conv_1(x2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)






class FSUNet(BaseModel):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(FSUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.lambda_l2 = 0.01

        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.loss1 = FocalLoss()
        self.loss2 = BCELoss()
        self.loss3 = EdgeLoss()

        self.gate = gsc.GatedSpatialConv2d(64, 64)

        self.encoder = encoder(
            mlp_ratio=2.0,
            embed_dim=64,
            depths=(1, 2, 8, 2),
            drop_path_rate=0.15,
            act_layer='RELU',
            fork_feat=True
        )
        self.h_gc1 = H_GC(filters[0])
        self.h_gc2 = H_GC(filters[1])
        self.h_gc3 = H_GC(filters[2])
        self.h_gc4 = H_GC(filters[3])

        self.l_sr1 = L_SR(filters[0])
        self.l_sr2 = L_SR(filters[1])
        self.l_sr3 = L_SR(filters[2])
        self.l_sr4 = L_SR(filters[3])

        self.gate1 = gsc.GatedSpatialConv2d(64, 64)

        self.gate2 = gsc.GatedSpatialConv2d(128, 128)

        self.gate3 = gsc.GatedSpatialConv2d(256, 256)

        self.gate4 = gsc.GatedSpatialConv2d(512, 512)


    def forward(self, x, gt=None, data_samples=None, mode="predict"):
        x0 = self.inc(x)  # 4 1 256 256

        assert not torch.isnan(self.inc[0].weight.any()), "conv weight 含 NaN"

        out = self.encoder(x0)  # 4 64 256 256
        X_1 = out[0]  # 4 64 256 256
        X_2 = out[1]  # 4 128 128 128
        X_3 = out[2]  # 4 256 64 64
        X_4 = out[3]  # 4 512 32 32

        f_low_1 = self.l_sr1(X_1)
        f_low_2 = self.l_sr2(X_2)
        f_low_3 = self.l_sr3(X_3)
        f_low_4 = self.l_sr4(X_4)

        f_high_1 = self.h_gc1(X_1)
        f_high_2 = self.h_gc2(X_2)
        f_high_3 = self.h_gc3(X_3)
        f_high_4 = self.h_gc4(X_4)

        T_1 = self.gate1(f_low_1, f_high_1)
        T_2 = self.gate2(f_low_2, f_high_2)
        T_3 = self.gate3(f_low_3, f_high_3)
        T_4 = self.gate4(f_low_4, f_high_4)

        f_high_2 = F.interpolate(f_high_2, scale_factor=2, mode='nearest')
        f_high_3 = F.interpolate(f_high_3, scale_factor=4, mode='nearest')
        f_high_4 = F.interpolate(f_high_4, scale_factor=8, mode='nearest')
        f_high = (f_high_1 + f_high_2 + f_high_3 + f_high_4) / 4

        m4 = self.up1(X_4, T_4)
        m3 = self.up2(m4, T_3)
        m2 = self.up3(m3, T_2)
        m1 = self.up4(m2, T_1)

        m1 = self.gate(m1, f_high)
        logits = self.outc(m1).sigmoid()

        l2_loss = 0
        for param in self.parameters():
            if param.requires_grad:  # 确保仅对可训练参数应用正则化
                l2_loss += torch.sum(param ** 2)

        # L2正则化系数
        lambda_l2 = 1e-3

        if mode == "predict":
            return logits
        else:
            gt = torch.clamp(gt, min=0, max=1)
            return {"loss": (self.loss1(logits, gt) + self.loss2(logits, gt)) * 10000 + self.loss3(f_high_4, gt) * 5000 + l2_loss * lambda_l2}

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)