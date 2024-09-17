import torch
import torch.nn as nn


class AudioEmbeddingModel(nn.Module):
    def __init__(self):
        super(AudioEmbeddingModel, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define max pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Use adaptive pooling to handle variable input lengths
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=64)
        self.relu = nn.ReLU()
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 64, 256)  # Now fixed due to adaptive pooling
        self.fc2 = nn.Linear(256, 128)  # Embedding size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x

