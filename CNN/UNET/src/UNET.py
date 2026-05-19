import torch
import torch.nn as nn
import torchvision.transforms.functional as F



# Basic Block 1: 2 - 3x3 convolutions followed by ReLu and batch-norm but with padding unlike the main paper to avoid cropping
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



# Basic Block 2: Encoder
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv   = DoubleConv(in_channels, out_channels)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip   = self.conv(x)   
        pooled = self.pool(skip)
        return skip, pooled



# Basic Block 3: Decoder where upsample > concatenate skip > DoubleConv
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,  kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.resize(x, size=skip.shape[2:])

        x = torch.cat([skip, x], dim=1)   
        return self.conv(x)



# Full UNet
# 4 encoder stages  (64 > 128 > 256 > 512)
# Bottleneck        (512 > 1024)
# 4 decoder stages  (1024 > 512 > 256 > 128 > 64)
# 1x1 conv output head
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, features=(64, 128, 256, 512)):
        super().__init__()

        # Encoder 
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # Bottle Neck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder 
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2      
        for f in reversed(features):
            self.decoders.append(DecoderBlock(ch, f))
            ch = f

        # Output 
        self.output_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass - consume skips in reverse order ─
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output head 
        return self.output_conv(x)



# Sanity check

if __name__ == "__main__":
    model  = UNet(in_channels=1, num_classes=2)
    x      = torch.randn(1, 1, 572, 572)  
    out    = model(x)
    print(f"Input  : {x.shape}")           
    print(f"Output : {out.shape}")        
    total  = sum(p.numel() for p in model.parameters())
    print(f"Params : {total:,}")