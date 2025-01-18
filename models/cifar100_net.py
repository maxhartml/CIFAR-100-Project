import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration.config_1 import DROPOUT_RATE

class CIFAR100Net(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed for classifying images in the CIFAR-100 dataset.
    The network consists of three convolutional blocks followed by three fully connected layers.
    """

    def __init__(self):
        super(CIFAR100Net, self).__init__()

        # ---------------------------------------------------
        # Convolutional Block 1
        # ---------------------------------------------------
        # First convolutional layer: input (3 channels, RGB), output (64 channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for stability
        
        # Second convolutional layer: input (64 channels), output (128 channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Pooling layer: downsample by a factor of 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x16x16

        # ---------------------------------------------------
        # Convolutional Block 2
        # ---------------------------------------------------
        # Third convolutional layer: input (128 channels), output (256 channels)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fourth convolutional layer: input (256 channels), output (256 channels)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling layer: downsample by a factor of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256x8x8

        # ---------------------------------------------------
        # Convolutional Block 3
        # ---------------------------------------------------
        # Fifth convolutional layer: input (256 channels), output (512 channels)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Sixth convolutional layer: input (512 channels), output (512 channels)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # Pooling layer: downsample by a factor of 2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 512x4x4

        # ---------------------------------------------------
        # Fully Connected Layers
        # ---------------------------------------------------
        # First fully connected layer: input (512x4x4 = 8192), output (1024)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)  # Dropout to reduce overfitting

        # Second fully connected layer: input (1024), output (512)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(DROPOUT_RATE) # Dropout to reduce overfitting

        # Third fully connected layer: input (512), output (100 classes for CIFAR-100)
        self.fc3 = nn.Linear(512, 100)

    def forward(self, x):
        """
        Define the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 100), representing class scores.
        """
        # Convolutional Block 1
        x = F.relu(self.bn1(self.conv1(x)))  # Apply conv1 -> batch norm -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))  # Apply conv2 -> batch norm -> ReLU
        x = self.pool1(x)  # Downsample

        # Convolutional Block 2
        x = F.relu(self.bn3(self.conv3(x)))  # Apply conv3 -> batch norm -> ReLU
        x = F.relu(self.bn4(self.conv4(x)))  # Apply conv4 -> batch norm -> ReLU
        x = self.pool2(x)  # Downsample

        # Convolutional Block 3
        x = F.relu(self.bn5(self.conv5(x)))  # Apply conv5 -> batch norm -> ReLU
        x = F.relu(self.bn6(self.conv6(x)))  # Apply conv6 -> batch norm -> ReLU
        x = self.pool3(x)  # Downsample

        # Flatten the tensor to prepare for fully connected layers
        x = torch.flatten(x, 1)  # Shape: (batch_size, 8192)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))  # Apply fc1 -> ReLU
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Apply fc2 -> ReLU
        x = self.dropout2(x)  # Apply dropout
        x = self.fc3(x)  # Final layer for class scores

        return x