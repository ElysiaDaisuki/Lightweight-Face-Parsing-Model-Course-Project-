import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for parameter efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                  padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.relu2(x)

class BasicResBlock(nn.Module):
    """Basic residual block using depthwise separable conv"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        
        shortcut_out = self.shortcut(identity)
        
        out = out + shortcut_out
        
        return out

class LightweightFaceParser(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.ModuleList([
            BasicResBlock(64, 64, stride=2),
            BasicResBlock(64, 128, stride=2),
            BasicResBlock(128, 256, stride=2),
            BasicResBlock(256, 512, stride=2)
        ])

        self.decoder = nn.ModuleList()

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(512 + 256, 256)
        ))

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(256 + 128, 128)
        ))

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(128 + 64, 64)
        ))

        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

        self._initialize_weights()
        params = self.count_parameters()
        print(f"Model parameters: {params:,}")
        print(f"Max allowed: 1,821,085")
        print(f"Within limit: {params <= 1821085}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input(x)
        skips = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            skips.append(x)

        for i, layer in enumerate(self.decoder):
            x = layer[0](x)
            x = torch.cat([x, skips[2 - i]], dim=1)
            x = layer[1](x)

        x = self.final_upsample(x)
        x = self.output(x)
        return x

def verify_model_size():
    model = LightweightFaceParser(num_classes=19)
    num_params = model.count_parameters()

    print("\n" + "="*50)
    print("Model Size Verification")
    print("="*50)
    print(f"Model parameters: {num_params:,}")
    print(f"Maximum allowed: 1,821,085")
    print(f"Within limit: {num_params <= 1821085}")
    print(f"Remaining: {1821085 - num_params:,}")
    print("="*50 + "\n")

    return model