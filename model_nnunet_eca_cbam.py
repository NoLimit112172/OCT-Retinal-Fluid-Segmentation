
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """nnU-Net style block: Conv -> InstanceNorm -> LeakyReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """nnU-Net style residual convolution block"""

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
        return self.act(out + identity)


class DownBlock(nn.Module):
    """Downsampling block with stride=2 convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class ECABlock(nn.Module):
    """
    ECA: Efficient Channel Attention.

    Used only after encoder outputs in this experiment.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y = self.avg_pool(x)                         # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)          # B,1,C
        y = self.conv(y)                             # B,1,C
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        refined = x * y
        return x + self.gamma * (refined - x)


class ChannelAttention(nn.Module):
    """CBAM channel attention"""

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
    """CBAM spatial attention"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(att))


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module.

    Used only after decoder fusion blocks in this experiment.
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


class UpBlock(nn.Module):
    """Upsampling + skip concat + residual fusion"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
            ])

        x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class nnUNet(nn.Module):
    """
    nnU-Net + ECA-Encoder + CBAM-Decoder.

    - ECA is applied to all encoder outputs: x1, x2, x3, x4, x5.
    - CBAM is applied after each decoder fusion block.
    - No SDA, no SASC, no ASPP.
    - base_channels defaults to 32.
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

        self.enc1 = ResidualConvBlock(n_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.enc5 = DownBlock(c4, c5)

        self.eca1 = ECABlock(c1)
        self.eca2 = ECABlock(c2)
        self.eca3 = ECABlock(c3)
        self.eca4 = ECABlock(c4)
        self.eca5 = ECABlock(c5)

        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)

        self.cbam1 = CBAM(c4)
        self.cbam2 = CBAM(c3)
        self.cbam3 = CBAM(c2)
        self.cbam4 = CBAM(c1)

        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.eca1(self.enc1(x))
        x2 = self.eca2(self.enc2(x1))
        x3 = self.eca3(self.enc3(x2))
        x4 = self.eca4(self.enc4(x3))
        x5 = self.eca5(self.enc5(x4))

        d1 = self.cbam1(self.up1(x5, x4))
        d2 = self.cbam2(self.up2(d1, x3))
        d3 = self.cbam3(self.up3(d2, x2))
        d4 = self.cbam4(self.up4(d3, x1))

        return self.out_conv(d4)


if __name__ == "__main__":
    model = nnUNet(n_channels=1, n_classes=4, base_channels=32)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eca_params = sum(p.numel() for n, p in model.named_parameters() if "eca" in n and p.requires_grad)
    cbam_params = sum(p.numel() for n, p in model.named_parameters() if "cbam" in n and p.requires_grad)

    print(f"total params: {total_params / 1e6:.2f} M")
    print(f"ECA params  : {eca_params / 1e6:.4f} M")
    print(f"CBAM params : {cbam_params / 1e6:.4f} M")
