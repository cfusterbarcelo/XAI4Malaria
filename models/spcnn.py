# models/spcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class SPCNN(BaseModel):
    """
    Soft Attention Parallel Convolutional Neural Network (SPCNN)
    for binary classification (e.g. malaria diagnosis).
    """

    def _build(self):
        self.parallel_kernels = [11, 9, 7, 5, 3]
        in_channels = self.input_shape[0]  # usually 3 (RGB)
        parallel_layers = []

        for k in self.parallel_kernels:
            layer = nn.Conv2d(in_channels, 16, kernel_size=k, padding=k // 2)
            parallel_layers.append(layer)

        self.parallel_convs = nn.ModuleList(parallel_layers)

        # Sequential convolutional layers with soft attention
        self.conv_blocks = nn.Sequential(
            ConvBNReLU(16 * len(self.parallel_kernels), 128, use_attention=True),
            ConvBNReLU(128, 64, use_attention=True),
            ConvBNReLU(64, 32, use_attention=True),
            ConvBNReLU(32, 16, use_attention=True)
        )

        # Dropout after last 2 conv layers
        self.dropout_conv1 = nn.Dropout(0.5)
        self.dropout_conv2 = nn.Dropout(0.5)

        # NEW: Adaptive pooling to enforce 2x2 final map
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Flatten + Z-score normalization
        self.flatten = nn.Flatten()
        self.norm = ZScoreNorm()

        # Fully connected layers
        self.fc1 = nn.Linear(2 * 2 * 16, 500)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, self.num_classes)

        # Final model modules
        self.feature_extractor = nn.Sequential(
            ParallelBlock(self.parallel_convs),
            self.conv_blocks,
            self.pool,            # << added
            self.dropout_conv1,
            self.dropout_conv2,
            self.flatten,
            self.norm             # << added
        )

        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.dropout_fc1,
            self.fc2,
            nn.ReLU(),
            self.fc3
        )


class SoftAttention(nn.Module):
    """
    Soft attention mechanism that generates an attention map and scales the input feature map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar γ

    def forward(self, x):
        attention = self.conv(x)
        attention = torch.softmax(attention.view(x.size(0), -1), dim=-1)
        attention = attention.view(x.size(0), 1, x.size(2), x.size(3))
        return x + self.gamma * (x * attention)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if use_attention:
            self.attention = SoftAttention(out_channels)

    def forward(self, x):
        x = self.block(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class ParallelBlock(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x):
        out = [conv(x) for conv in self.conv_layers]
        return torch.cat(out, dim=1)


class ZScoreNorm(nn.Module):
    """
    Applies z-score normalization across the channel dimension (features).
    """
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - mean) / std
