import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN model for hand gesture recognition with three 3x3 convolutional layers
    """

    def __init__(self, num_classes=10, hidden_size=128, dropout_rate=0.5):
        super(CNN, self).__init__()

        # Conv layers with batch norm and max pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)  # Regular dropout for FC layers

        # Calculate size after convolutions and pooling
        # Input: 64x64 -> After 3 pooling layers: 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


def get_model(num_classes=10, hidden_size=128, dropout_rate=0.5):
    """
    Factory function to create a model instance

    Args:
        num_classes (int): Number of output classes
        hidden_size (int): Size of hidden layer
        dropout_rate (float): Dropout probability

    Returns:
        model: Instantiated model
    """
    return CNN(
        num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate
    )


def count_parameters(model):
    """
    Count the number of trainable parameters in the model

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
