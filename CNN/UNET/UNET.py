import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────
# Building block: two 3×3 convolutions (same-padding, no bias)
# followed by BatchNorm + ReLU each.
# The original paper used valid (no) padding — we use padding=1
# to keep spatial size the same, which is the modern convention.
# ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────
# Encoder block: DoubleConv → MaxPool
# Returns both the feature map (for the skip connection)
# and the pooled output (to pass deeper into the encoder).
# ─────────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv   = DoubleConv(in_channels, out_channels)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip   = self.conv(x)   # kept for skip connection
        pooled = self.pool(skip)
        return skip, pooled


# ─────────────────────────────────────────────────────────────
# Decoder block: upsample → concatenate skip → DoubleConv
#
# Upsampling is done with ConvTranspose2d (learnable), matching
# the paper.  If the encoder used valid padding the spatial sizes
# will not align perfectly; we centre-crop the skip connection
# to match (same strategy used in the original paper).
# ─────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # up-conv 2×2 halves the channel count
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle potential size mismatch caused by odd-dimension inputs
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:])

        x = torch.cat([skip, x], dim=1)   # channel-wise concatenation
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# Full UNet
#   - 4 encoder stages  (64 → 128 → 256 → 512)
#   - Bottleneck        (512 → 1024)
#   - 4 decoder stages  (1024 → 512 → 256 → 128 → 64)
#   - 1×1 conv output head
# ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2,
                 features=(64, 128, 256, 512)):
        """
        Args:
            in_channels : number of input image channels (1 for grayscale,
                          3 for RGB)
            num_classes : number of segmentation classes (output channels)
            features    : channel progression through the encoder
        """
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # ── Bottleneck ────────────────────────────────────────
        # Deepest part of the U — no pooling, just DoubleConv
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Decoder ───────────────────────────────────────────
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2        # start from bottleneck channels
        for f in reversed(features):
            # DecoderBlock takes in_channels=ch, produces out_channels=f
            self.decoders.append(DecoderBlock(ch, f))
            ch = f

        # ── Output head ───────────────────────────────────────
        # 1×1 conv: maps 64 feature channels → num_classes
        self.output_conv = nn.Conv2d(features[0], num_classes,
                                     kernel_size=1)

    def forward(self, x):
        # ── Encoder pass — store skip connections ────────────
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # ── Bottleneck ────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decoder pass — consume skips in reverse order ────
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # ── Output head ───────────────────────────────────────
        return self.output_conv(x)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = UNet(in_channels=1, num_classes=2)
    x      = torch.randn(1, 1, 572, 572)   # paper's input size
    out    = model(x)
    print(f"Input  : {x.shape}")           # (1, 1, 572, 572)
    print(f"Output : {out.shape}")          # (1, 2, 572, 572) with padding=1
    total  = sum(p.numel() for p in model.parameters())
    print(f"Params : {total:,}")