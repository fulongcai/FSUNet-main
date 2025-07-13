import copy
from functools import partial
from typing import List
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor
import torch.nn.functional as F
from .wavelet import DWT
import os

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3_pre = dim // n_div
        self.dim_conv3_rear = dim - self.dim_conv3_pre * (n_div - 2)
        self.dim_untouched = dim - self.dim_conv3_pre - self.dim_conv3_rear
        self.partial_conv3_pre = nn.Conv2d(self.dim_conv3_pre, self.dim_conv3_pre, 3, 1, 1,bias=False)
        self.partial_conv3_rear = nn.Conv2d(self.dim_conv3_rear, self.dim_conv3_rear, 3, 1, 1, bias=False)
        self.partial_conv3_pre_new = nn.Conv2d(self.dim_conv3_pre, self.dim_conv3_pre, 3, 1, 1,bias=False)
        self.partial_conv3_rear_new = nn.Conv2d(self.dim_conv3_rear, self.dim_conv3_rear, 3, 1, 1, bias=False)

        self.conv = nn.Conv2d(dim, dim,3,1,1,bias=False)
        self.dwt = DWT()
        # 低频引导分支
        self.low_guidance = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.Sigmoid()
        )
        # 高频引导分支
        self.high_guidance = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 1 // 2, 3, padding=1),  # HL+LH+HH
            nn.Sigmoid()
        )
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        coeffs = self.dwt(x)
        c = coeffs.size(1)
        # 分离子带
        LL = coeffs[:, :c // 4, :, :]
        HL = coeffs[:, c // 4:c // 2, :, :]
        LH = coeffs[:, c // 2:c * 3 // 4, :, :]
        HH = coeffs[:, c * 3 // 4:, :, :]
        LL = torch.abs(LL)
        LH = torch.abs(LH)
        HL = torch.abs(HL)
        HH = torch.abs(HH)

        A_low = F.interpolate(self.low_guidance(LL), scale_factor=2, mode='nearest')
        A_high = F.interpolate(self.high_guidance(torch.cat([HL, LH, HH], dim=1)), scale_factor=2, mode='nearest')
        x1, x2, x3 = torch.split(x, [self.dim_conv3_pre, self.dim_untouched, self.dim_conv3_rear], dim=1)
        x1 = x1 * A_low
        x3 = x3 * A_high
        x_low = self.partial_conv3_pre(x1)
        x_high = self.partial_conv3_rear(x3)
        x_split = torch.cat((x_low, x2, x_high), 1)

        return x_split

class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x

class encoder(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 embed_dim=64,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        self.init__ = super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.model_wcs = []

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                            n_div=n_div,
                            depth=depths[i_stage],
                            mlp_ratio=self.mlp_ratio,
                            drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                            layer_scale_init_value=layer_scale_init_value,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            pconv_fw_type=pconv_fw_type
                            )
            # patch merging layer
            if i_stage != 0:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** (i_stage - 1)),
                                 norm_layer=norm_layer)
                )
            stages_list.append(stage)

        self.W_SDs = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_det(self, x: Tensor) -> Tensor:
        outs = []
        for idx, stage in enumerate(self.W_SDs):  # W_SDs为多层W_SD模块叠加
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs