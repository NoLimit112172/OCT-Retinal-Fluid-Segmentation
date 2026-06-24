import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """nnU-Net 风格基础块: Conv -> InstanceNorm -> LeakyReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
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
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
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
            ResidualConvBlock(out_channels, out_channels)
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
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class SDA(nn.Module):
    """
    SDA: Self-Adaptive Dual-Attention

    作用位置：
    - 只作用于 skip connection 的 encoder 特征；
    - 不作用于 bottleneck；
    - 不改变 decoder 结构；
    - 用于验证 SDA 单独作用。

    设计说明：
    - 包含像素/空间关系注意力和通道关系注意力；
    - 先对特征下采样后计算注意力，降低显存开销；
    - 再插值回原空间尺寸；
    - 使用 gamma 初始值 0.1 的残差形式，让训练初期更稳定。
    """

    def __init__(self, in_channels, pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.shape

        if self.pool_size > 1:
            xd = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        else:
            xd = x

        hp, wp = xd.shape[2], xd.shape[3]
        n = hp * wp

        xr = xd.view(b, c, n)          # B, C, N
        xr_t = xr.permute(0, 2, 1)     # B, N, C

        # Pixel / spatial attention: N x N
        energy_pixel = torch.bmm(xr_t, xr)
        att_pixel = F.softmax(energy_pixel / max(n ** 0.5, 1.0), dim=-1)
        x_att_pixel_d = torch.bmm(att_pixel, xr_t).permute(0, 2, 1).contiguous()
        x_att_pixel_d = x_att_pixel_d.view(b, c, hp, wp)
        x_att_pixel = F.interpolate(x_att_pixel_d, size=(h, w), mode='bilinear', align_corners=False)

        # Channel attention: C x C
        energy_channel = torch.bmm(xr, xr_t)
        att_channel = F.softmax(energy_channel / max(c ** 0.5, 1.0), dim=-1)
        x_att_channel_d = torch.bmm(att_channel, xr).view(b, c, hp, wp)
        x_att_channel = F.interpolate(x_att_channel_d, size=(h, w), mode='bilinear', align_corners=False)

        enhanced = x + 0.5 * (self.alpha * x_att_pixel + self.beta * x_att_channel)

        # 稳定残差：初始时仅小幅引入 SDA 增强特征
        return x + self.gamma * (enhanced - x)


class nnUNet(nn.Module):
    """
    实验 5：nnU-Net + SDA

    模块位置：
    - SDA 只放在 skip connection 上；
    - 编码器输出 x1/x2/x3/x4 在送入 decoder concat 之前经过 SDA；
    - bottleneck x5 不加 SDA；
    - decoder 不加 SASC / CBAM / ASPP；
    - 用于验证 SDA 模块单独作用。

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

        # SDA only on skip connections
        # 256 -> pool 8 -> 32, 128 -> pool 4 -> 32, 64 -> pool 2 -> 32, 32 -> pool 1 -> 32
        self.sda1 = SDA(c1, pool_size=8)
        self.sda2 = SDA(c2, pool_size=4)
        self.sda3 = SDA(c3, pool_size=2)
        self.sda4 = SDA(c4, pool_size=1)

        # Decoder
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

        # SDA-enhanced skip features
        s1 = self.sda1(x1)
        s2 = self.sda2(x2)
        s3 = self.sda3(x3)
        s4 = self.sda4(x4)

        d1 = self.up1(x5, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)

        return self.out_conv(d4)


if __name__ == '__main__':
    model = nnUNet(n_channels=1, n_classes=4)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print('input :', x.shape)
    print('output:', y.shape)
