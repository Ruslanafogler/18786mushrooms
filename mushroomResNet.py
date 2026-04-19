
#uses the residual block pattern from He et al. (2016). 
#key is the skip connection: instead of learning a mapping H(x), each
#block learns a residual* F(x) = H(x) - x so the output is x + F(x).
#makes it easy for a block to learn the identity (just drive F to zero),
#which solves the vanishign gradient problems 

# Two block types:
# BasicBlock which has two 3×3 convs (ResNet-18 / 34 style)
# Bottleneck which as 1×1 → 3×3 → 1×1 convs (ResNet-50+ style, optional)


import torch
import torch.nn as nn



class BasicBlock(nn.Module):
    # Two-layer residual block (ResNet-18 / 34):
    #note that BN is batch norm
    # x to Conv3×3 to BN to ReLU to  Conv3×3 to BN ──> (+) → ReLU
    # │                                                 |
    # └──────────── identity                      ──────┘

    expansion = 1  # output channels = base channels × expansion

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut: identity when dims match, 1×1 conv otherwise
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out = out + identity #behold the resdiual rirght here 
        return self.relu(out)


class Bottleneck(nn.Module):
    # Three-layer bottleneck block (ResNet-50 / 101 / 152):

    #     x tp 1×1 to BN to ReLU to 3×3 to BN to ReLU to 1×1 to BN ──> (+) > ReLU
    #     │                                                          |
    #     └──────────────── identity                   ──────────────┘

    #The 1×1 convs compress then expand the channel dimension, 
    #makes the expensive 3×3 conv operate on fewer channels.  Output channels =
    #base channels × 4 (the expansion factor).
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        # out_channels here is the "base" width; final output is base × 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu  = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        expanded = out_channels * self.expansion
        if stride != 1 or in_channels != expanded:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(expanded),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))

        out = out + identity
        return self.relu(out)


# Named presets: (block_type, layer_counts)
RESNET_PRESETS = {
    "resnet18":  (BasicBlock,  [2, 2, 2, 2]),
    "resnet34":  (BasicBlock,  [3, 4, 6, 3]),
    "resnet50":  (Bottleneck,  [3, 4, 6, 3]),
}


class MushroomResNet(nn.Module):

    #Architecture is:

    #Conv7×7/2 to BN to ReLU to MaxPool3×3/2
    #Stage 1 (n blocks @ 64 ch)
    #Stage 2 (n blocks @ 128 ch, stride-2 entry)
    #Stage 3 (n blocks @ 256 ch, stride-2 entry)
    #Stage 4 (n blocks @ 512 ch, stride-2 entry)
    #AdaptiveAvgPool to Dropout to FC(num_classes)

    def __init__(
        self,
        preset:        str   = "resnet18",
        base_width:    int   = 64,
        block_dropout: float = 0.0,
        head_dropout:  float = 0.5,
        num_classes:   int   = 2,
        input_channels: int  = 3,
    ):
        super().__init__()

        if preset not in RESNET_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. "
                             f"Choose from {list(RESNET_PRESETS)}")

        block_cls, layer_counts = RESNET_PRESETS[preset]

        self.hyperparams = dict(
            preset=preset,
            base_width=base_width,
            block_dropout=block_dropout,
            head_dropout=head_dropout,
            num_classes=num_classes,
            input_channels=input_channels,
        )

        self._in_channels = base_width

        #stem: aggressive spatial reduction before the residual stages
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_width, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        #four residual stages
        widths = [base_width * (2 ** i) for i in range(4)]   # 64, 128, 256, 512
        self.layer1 = self._make_stage(block_cls, widths[0],
                                       layer_counts[0], stride=1,
                                       dropout=block_dropout)
        self.layer2 = self._make_stage(block_cls, widths[1],
                                       layer_counts[1], stride=2,
                                       dropout=block_dropout)
        self.layer3 = self._make_stage(block_cls, widths[2],
                                       layer_counts[2], stride=2,
                                       dropout=block_dropout)
        self.layer4 = self._make_stage(block_cls, widths[3],
                                       layer_counts[3], stride=2,
                                       dropout=block_dropout)

        # Expose the full conv backbone as `self.features` for Grad-CAM 
        self.features = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

        #classification head
        final_channels = widths[3] * block_cls.expansion
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(final_channels, num_classes),
        )

        self._init_weights()

    def _make_stage(self, block_cls, base_ch, num_blocks, stride, dropout):
        layers = [block_cls(self._in_channels, base_ch,
                            stride=stride, dropout=dropout)]
        self._in_channels = base_ch * block_cls.expansion
        for _ in range(1, num_blocks):
            layers.append(block_cls(self._in_channels, base_ch,
                                    stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        #Kaiming for conv layers, constant init for BN.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)
