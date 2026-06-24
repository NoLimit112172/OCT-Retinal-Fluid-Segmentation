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
    """上采样 + skip concat + 残差融合"""

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


class ChannelAttention(nn.Module):
    """
    CBAM 的通道注意力分支。
    使用 AvgPool + MaxPool 两个描述子，经过共享 MLP 得到通道权重。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    CBAM 的空间注意力分支。
    沿通道维度做 average / max，拼接后用 7x7 卷积得到空间权重。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attention))


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module

    本实验中 CBAM 放在 decoder block 后面，而不是放在 skip connection 上。
    为了训练更稳定，这里采用轻微残差形式：
        output = x + gamma * (CBAM(x) - x)
    初始 gamma=0.1，避免一开始对特征扰动过大。
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        refined = x * self.channel_attention(x)
        refined = refined * self.spatial_attention(refined)
        return x + self.gamma * (refined - x)


class nnUNet(nn.Module):
    """
    实验 4：nnU-Net + CBAM-Decoder

    模块位置：
    - CBAM 只放在 Decoder 阶段；
    - 每次 UpBlock 完成上采样、concat、卷积融合之后，再经过 CBAM 做空间-通道注意力细化；
    - Encoder、Bottleneck 和 Skip Connection 都保持 nnU-Net Base 原样；
    - 便于和 ECA-Encoder、SDA/SASC 等实验做公平对比。

    注意：这是用于毕设消融实验的 2D nnU-Net 风格 backbone，
    不是官方 nnU-Net 框架的完整复现。
    """

    def __init__(self, n_channels=1, n_classes=4, base_channels=32):
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

        # Decoder
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)

        # Decoder 后处理注意力：不作用于 skip connection
        self.cbam1 = CBAM(c4)
        self.cbam2 = CBAM(c3)
        self.cbam3 = CBAM(c2)
        self.cbam4 = CBAM(c1)

        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d1 = self.up1(x5, x4)
        d1 = self.cbam1(d1)

        d2 = self.up2(d1, x3)
        d2 = self.cbam2(d2)

        d3 = self.up3(d2, x2)
        d3 = self.cbam3(d3)

        d4 = self.up4(d3, x1)
        d4 = self.cbam4(d4)

        return self.out_conv(d4)


if __name__ == "__main__":
    model = nnUNet(n_channels=1, n_classes=4)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
