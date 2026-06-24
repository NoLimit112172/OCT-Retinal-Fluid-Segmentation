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
    """残差卷积块：保持与你原 nnU-Net + SDA/SASC 版本一致。"""

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
    """stride=2 下采样。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """上采样 + 原始 skip concat + 残差融合。"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # 防止输入尺寸不是 2 的整数幂时，上采样后和 skip 特征图尺寸不完全一致。
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


class NNUNetBase(nn.Module):
    """
    nnU-Net Base：不加入 SDA、SASC、ECA、CBAM 等额外模块。

    注意：这是用于毕设消融实验的 2D nnU-Net 风格 backbone，
    """

    def __init__(self, n_channels=1, n_classes=4, base_channels=16):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 10  # 320 when base_channels=32

        # Encoder
        self.enc1 = ResidualConvBlock(n_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.enc5 = DownBlock(c4, c5)

        # Decoder：直接使用原始 skip features，不加任何增强模块。
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)

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


# 兼容旧代码写法：如果你还想 from model_xxx import nnUNet，也能正常跑。
nnUNet = NNUNetBase


if __name__ == "__main__":
    model = NNUNetBase(n_channels=1, n_classes=4)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
