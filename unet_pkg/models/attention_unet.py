"""
unet_pkg.models.attention_unet
带 Attention Gate 的 U-Net。
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(channels: int, norm_type: str = "batch", num_groups: int = 8) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        groups = min(num_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm_type: str = "batch",
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # 旧实现（保留）:
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        # )
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            make_norm(mid_channels, norm_type=norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels, norm_type=norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module):
        super().__init__()
        # 旧实现（保留）:
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     DoubleConv(in_channels, out_channels),
        # )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            block(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def init_weights(m, init_type='kaiming'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class GridAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 mode='concatenation', sub_sample_factor=(2, 2)):
        super().__init__()
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = (sub_sample_factor, sub_sample_factor)

        self.mode = mode
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels if inter_channels is not None else max(in_channels // 2, 1)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels),
        )

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=self.sub_sample_kernel_size,
                               stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                             kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.children():
            init_weights(m, init_type='kaiming')

        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        else:
            self.operation_function = self._concatenation_residual

    def forward(self, x, g):
        return self.operation_function(x, g)

    def _concatenation(self, x, g):
        input_size = x.size()
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.softplus(theta_x + phi_g)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x_size[2:])
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f


class UnetGridGatingSignal2D(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), is_batchnorm=True, norm_type: str = "batch"):
        super().__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, stride=(1, 1), padding=(0, 0), bias=False),
                make_norm(out_size, norm_type=norm_type),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, stride=(1, 1), padding=(0, 0), bias=False),
                nn.ReLU(inplace=True),
            )
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        return self.conv1(inputs)


class ResidualConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm_type: str = "batch",
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(mid_channels, norm_type=norm_type)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(out_channels, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels, norm_type=norm_type),
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ASPPContext(nn.Module):
    def __init__(self, channels: int, norm_type: str = "batch"):
        super().__init__()
        dilations = [1, 2, 4]
        branches = []
        for d in dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    make_norm(channels, norm_type=norm_type),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1, bias=False),
            make_norm(channels, norm_type=norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        bilinear=True,
        use_attention=True,
        attention_mode='concatenation',
        attention_dsample=(2, 2),
        base_channels: int = 64,
        block_type: str = "double_conv",
        norm_type: str = "batch",
        bottleneck_context: bool = False,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        self.deep_supervision = deep_supervision
        self.bottleneck_context = bottleneck_context

        block_cls = DoubleConv if block_type == "double_conv" else ResidualConv
        block = lambda in_ch, out_ch, mid_ch=None: block_cls(  # noqa: E731
            in_ch, out_ch, mid_ch, norm_type=norm_type
        )

        # 旧实现（保留）:
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 8
        factor = 2 if bilinear else 1
        ch5 = (base_channels * 16) // factor

        self.inc = block(n_channels, ch1)
        self.down1 = Down(ch1, ch2, block=block)
        self.down2 = Down(ch2, ch3, block=block)
        self.down3 = Down(ch3, ch4, block=block)
        self.down4 = Down(ch4, ch5, block=block)

        self.context = ASPPContext(ch5, norm_type=norm_type) if bottleneck_context else nn.Identity()

        if self.use_attention:
            self.gating = UnetGridGatingSignal2D(
                in_size=ch5,
                out_size=ch4,
                kernel_size=(1, 1),
                is_batchnorm=True,
                norm_type=norm_type,
            )
            self.attentionblock1 = GridAttentionBlock2D(ch1, ch4, max(ch1 // 2, 1), attention_mode, attention_dsample)
            self.attentionblock2 = GridAttentionBlock2D(ch2, ch4, max(ch2 // 2, 1), attention_mode, attention_dsample)
            self.attentionblock3 = GridAttentionBlock2D(ch3, ch4, max(ch3 // 2, 1), attention_mode, attention_dsample)
            self.attentionblock4 = GridAttentionBlock2D(ch4, ch4, max(ch4 // 2, 1), attention_mode, attention_dsample)

        # 旧实现（保留）:
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.up1 = Up(ch5 + ch4, ch4 // factor, block=block, bilinear=bilinear)
        self.up2 = Up(ch4, ch3 // factor, block=block, bilinear=bilinear)
        self.up3 = Up(ch3, ch2 // factor, block=block, bilinear=bilinear)
        self.up4 = Up(ch2, ch1, block=block, bilinear=bilinear)
        self.outc = OutConv(ch1, n_classes)

        if self.deep_supervision:
            self.ds_heads = nn.ModuleList([
                OutConv(ch4 // factor, n_classes),
                OutConv(ch3 // factor, n_classes),
                OutConv(ch2 // factor, n_classes),
            ])
        else:
            self.ds_heads = nn.ModuleList()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.context(x5)

        if self.use_attention:
            gating = self.gating(x5)
            g_x4, _ = self.attentionblock4(x4, gating)
            g_x3, _ = self.attentionblock3(x3, gating)
            g_x2, _ = self.attentionblock2(x2, gating)
            g_x1, _ = self.attentionblock1(x1, gating)
        else:
            g_x4, g_x3, g_x2, g_x1 = x4, x3, x2, x1

        x = self.up1(x5, g_x4)
        ds_features: List[torch.Tensor] = [x]
        x = self.up2(x, g_x3)
        ds_features.append(x)
        x = self.up3(x, g_x2)
        ds_features.append(x)
        x = self.up4(x, g_x1)
        logits = self.outc(x)
        # 旧实现（保留）:
        # return logits
        if self.deep_supervision and self.training:
            aux_logits = []
            for feature, head in zip(ds_features, self.ds_heads):
                aux = head(feature)
                aux = F.interpolate(aux, size=logits.shape[2:], mode="bilinear", align_corners=False)
                aux_logits.append(aux)
            return {"logits": logits, "aux_logits": aux_logits}
        return {"logits": logits}

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
