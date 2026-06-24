import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    SegNet Encoder Block:
    多层卷积 + MaxPool(return_indices=True)
    """

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()

        layers = []
        ch = in_channels
        for _ in range(num_convs):
            layers.append(ConvBNReLU(ch, out_channels))
            ch = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        size = x.size()
        x, indices = self.pool(x)
        return x, indices, size


class DecoderBlock(nn.Module):
    """
    SegNet Decoder Block:
    MaxUnpool + 多层卷积
    """

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        layers = []
        ch = in_channels
        for i in range(num_convs):
            next_ch = out_channels if i == num_convs - 1 else in_channels
            layers.append(ConvBNReLU(ch, next_ch))
            ch = next_ch

        self.conv = nn.Sequential(*layers)

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.conv(x)
        return x


class SegNet(nn.Module):
    """
    SegNet baseline for OCT retinal fluid segmentation.

    输入:
        [B, 1, H, W]
    输出:
        [B, n_classes, H, W]

    作为较基础的 encoder-decoder 基线：
    - 无 U-Net 式 skip feature concat；
    - 无 attention；
    - 无 ASPP；
    - 通过 MaxPool indices + MaxUnpool 恢复空间结构。
    """

    def __init__(self, n_channels=1, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enc1 = EncoderBlock(n_channels, 64, num_convs=2)
        self.enc2 = EncoderBlock(64, 128, num_convs=2)
        self.enc3 = EncoderBlock(128, 256, num_convs=3)
        self.enc4 = EncoderBlock(256, 512, num_convs=3)

        self.dec4 = DecoderBlock(512, 256, num_convs=3)
        self.dec3 = DecoderBlock(256, 128, num_convs=3)
        self.dec2 = DecoderBlock(128, 64, num_convs=2)
        self.dec1 = DecoderBlock(64, 64, num_convs=2)

        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x, idx1, size1 = self.enc1(x)
        x, idx2, size2 = self.enc2(x)
        x, idx3, size3 = self.enc3(x)
        x, idx4, size4 = self.enc4(x)

        x = self.dec4(x, idx4, size4)
        x = self.dec3(x, idx3, size3)
        x = self.dec2(x, idx2, size2)
        x = self.dec1(x, idx1, size1)

        logits = self.classifier(x)
        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    model = SegNet(n_channels=1, n_classes=4).to(device)
    model.eval()

    x = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        y = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.2f} M")
