import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration.config_1 import DROPOUT_RATE


class CNN(nn.Module):
    """
    Improved Convolutional Neural Network for CIFAR-100.
    Utilises reduced fully connected layers, global average pooling,
    and advanced activation functions.
    """
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x16x16

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256x8x8

        # Convolutional Block 3
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 512x4x4

        # Global Average Pooling instead of fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: 512x1x1

        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        # Convolutional Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # Leaky ReLU for better gradient flow
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Convolutional Block 2
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Convolutional Block 3
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)  # Output: (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 512)

        # Fully Connected Layers
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x