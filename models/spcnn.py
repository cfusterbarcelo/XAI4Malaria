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

        # Sequential convolutional layers
        self.conv_blocks = nn.Sequential(
            ConvBNReLU(16 * len(self.parallel_kernels), 128),
            ConvBNReLU(128, 64),
            ConvBNReLU(64, 32),
            ConvBNReLU(32, 16)
        )

        # Dropout after last 2 conv layers
        self.dropout_conv1 = nn.Dropout(0.5)
        self.dropout_conv2 = nn.Dropout(0.5)

        # Feature map size is assumed small enough for flatten
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(2 * 2 * 16, 500)  # adjust if needed
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, self.num_classes)

        # Final model modules
        self.feature_extractor = nn.Sequential(
            ParallelBlock(self.parallel_convs),
            self.conv_blocks,
            self.dropout_conv1,
            self.dropout_conv2,
            self.flatten
        )

        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.dropout_fc1,
            self.fc2,
            nn.ReLU(),
            self.fc3
        )


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class ParallelBlock(nn.Module):
    def __init__(self, conv_layers):
        """
        conv_layers: list or nn.ModuleList of conv layers to apply in parallel
        """
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x):
        out = [conv(x) for conv in self.conv_layers]
        return torch.cat(out, dim=1)  # concatenate along channel dimension
