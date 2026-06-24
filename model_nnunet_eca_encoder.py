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
    """更接近 nnU-Net 常用风格的残差卷积块"""

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
    """stride=2 下采样，更贴近 nnU-Net 风格"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """上采样 + skip concat + 残差卷积融合"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
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

        x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class ECALayer(nn.Module):
    """
    ECA: Efficient Channel Attention

    放在编码器输出后，用轻量级 1D 卷积建模局部通道交互。
    这里采用标准形式: GlobalAvgPool -> Conv1d -> Sigmoid -> channel re-weighting。
    """

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)              # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)                  # [B, 1, C]
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


class nnUNetECAEncoder(nn.Module):
    """
    实验 2：nnU-Net + ECA-Encoder

    与 nnU-Net Base 保持同一主体结构，只在编码器阶段加入 ECA 通道注意力模块。
    注意：这里没有 SDA/SASC，也没有 ASPP/CBAM，便于做单因素对比实验。
    """

    def __init__(self, n_channels=1, n_classes=4, base_channels=32, eca_kernel_size=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 10  # base_channels=32 时为 320

        # Encoder
        self.enc1 = ResidualConvBlock(n_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.enc5 = DownBlock(c4, c5)

        # ECA modules on encoder features
        # 为了保持“编码器增强”的定义，ECA 放在各编码器输出后。
        self.eca1 = ECALayer(c1, k_size=eca_kernel_size)
        self.eca2 = ECALayer(c2, k_size=eca_kernel_size)
        self.eca3 = ECALayer(c3, k_size=eca_kernel_size)
        self.eca4 = ECALayer(c4, k_size=eca_kernel_size)
        self.eca5 = ECALayer(c5, k_size=eca_kernel_size)

        # Decoder
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.eca1(self.enc1(x))
        x2 = self.eca2(self.enc2(x1))
        x3 = self.eca3(self.enc3(x2))
        x4 = self.eca4(self.enc4(x3))
        x5 = self.eca5(self.enc5(x4))

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        return self.out_conv(d4)


# 保持和之前训练脚本相近的导入习惯：from model_xxx import nnUNet
nnUNet = nnUNetECAEncoder


if __name__ == '__main__':
    model = nnUNet(n_channels=1, n_classes=4)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print('input :', x.shape)
    print('output:', y.shape)
