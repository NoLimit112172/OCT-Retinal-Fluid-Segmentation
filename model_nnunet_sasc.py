import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """nnU-Net 风格基础块: Conv -> InstanceNorm -> LeakyReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """nnU-Net 风格残差卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        )
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


class DownBlock(nn.Module):
    """stride=2 下采样"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class SASC(nn.Module):
    """
    SASC: Skip Adaptive Selective Calibration

    本文件是“nnU-Net + SASC”单独消融版本：
    - 不加入 SDA；
    - 不加入 ECA / CBAM / ASPP；
    - 只在 decoder 上采样后、skip concat 前，用 decoder_up 引导 skip feature 做选择性校准。

    作用逻辑：
    skip feature 表示 encoder 的浅层/中层细节信息；
    decoder_up 表示 decoder 上采样后的语义信息；
    SASC 根据两者共同生成通道门控和空间门控，对 skip feature 进行校准后再 concat。
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 8)

        # 通道选择：由 skip + decoder_up 的全局上下文共同决定
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 空间选择：由 skip 和 decoder_up 的 avg/max 响应共同决定
        padding = spatial_kernel // 2
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=spatial_kernel, padding=padding, bias=False),
            nn.Sigmoid(),
        )

        # 局部细化：让被筛选后的 skip 具有局部连续性
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # 残差门控，初始扰动较小，避免训练初期破坏原始 skip
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, skip, decoder_up):
        if skip.shape[2:] != decoder_up.shape[2:]:
            decoder_up = F.interpolate(
                decoder_up,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        joint = torch.cat([skip, decoder_up], dim=1)

        # B, C, 1, 1
        ch_gate = self.channel_gate(F.adaptive_avg_pool2d(joint, output_size=1))

        # B, 1, H, W
        skip_avg = torch.mean(skip, dim=1, keepdim=True)
        skip_max, _ = torch.max(skip, dim=1, keepdim=True)
        dec_avg = torch.mean(decoder_up, dim=1, keepdim=True)
        dec_max, _ = torch.max(decoder_up, dim=1, keepdim=True)
        sp_gate = self.spatial_gate(torch.cat([skip_avg, skip_max, dec_avg, dec_max], dim=1))

        # 增强式校准，而不是强行压低 skip。
        # gate 初始约为 0.25~0.5 时，输出仍接近原始 skip，训练更稳。
        gate = ch_gate * sp_gate
        enhanced = skip + self.gamma * skip * gate
        refined = self.refine(enhanced)

        return skip + self.gamma * (refined - skip)


class UpBlockSASC(nn.Module):
    """上采样 -> SASC 校准 skip -> concat -> 残差融合"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        if out_channels != skip_channels:
            self.align_decoder = nn.Conv2d(out_channels, skip_channels, kernel_size=1, bias=False)
        else:
            self.align_decoder = nn.Identity()

        self.sasc = SASC(skip_channels)
        self.fuse = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        decoder_for_gate = self.align_decoder(x)
        skip = self.sasc(skip, decoder_for_gate)

        x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class nnUNet(nn.Module):
    """
    实验：nnU-Net + SASC

    结构：
    Encoder
      ↓
    Bottleneck
      ↓
    Decoder upsample
      ↓
    SASC(skip, decoder_up)
      ↓
    concat + residual fusion
      ↓
    Output

    注意：
    - base_channels 默认仍为 32，不是 16；
    - 这是 SASC 单独消融，不包含 SDA。
    """

    def __init__(self, n_channels=1, n_classes=4, base_channels=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 10

        # Encoder
        self.enc1 = ResidualConvBlock(n_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.enc5 = DownBlock(c4, c5)

        # Decoder with SASC only
        self.up1 = UpBlockSASC(c5, c4, c4)
        self.up2 = UpBlockSASC(c4, c3, c3)
        self.up3 = UpBlockSASC(c3, c2, c2)
        self.up4 = UpBlockSASC(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        return self.out_conv(d4)


if __name__ == "__main__":
    model = nnUNet(n_channels=1, n_classes=4, base_channels=32)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sasc_params = sum(p.numel() for n, p in model.named_parameters() if "sasc" in n and p.requires_grad)
    print(f"total params: {total_params / 1e6:.2f} M")
    print(f"SASC params : {sasc_params / 1e6:.2f} M")
