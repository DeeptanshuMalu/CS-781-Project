import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class CarDetectorModel(nn.Module):
    def __init__(self, n_classes=2, img_size=32):
        super(CarDetectorModel, self).__init__()

        # Network parameters (matching TF model)
        self.img_size = img_size
        self.num_channels = 3
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.num_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Calculate the size after conv and pooling layers
        # Input: 32x32, after 3 max pools: 4x4, final feature maps: 32
        self.fc_input_size = 4*4*32
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        # Conv layer 1 + MaxPool + ReLU (matching TF createConvolutionalLayer)
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(x)

        # Conv layer 2 + MaxPool + ReLU
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(x)

        # Conv layer 3 + MaxPool + ReLU
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(x)

        # Flatten layer (matching TF createFlattenLayer)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1)

        # FC layer 1 + ReLU (matching TF createFcLayer with useRelu=True)
        x = self.fc1(x)
        x = F.relu(x)

        # FC layer 2 (matching TF createFcLayer with useRelu=False)
        x = self.fc2(x)

        return x