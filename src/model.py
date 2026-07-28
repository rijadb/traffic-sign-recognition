import torch
import torch.nn as nn


class TrafficSignCNN(nn.Module):

    def __init__(self):
        super().__init__()

        #conv2d = scans entire image 
        self.conv1 = nn.Conv2d(
            in_channels=3,      # 3 input channels (rgb)
            out_channels=32,    
            kernel_size=3,      # 3x3 pixels window
            padding=1           # keeps the size
        )

        # additional features
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # reducing image size while keeping strongest feature
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # activation function, removes negative values
        self.relu = nn.ReLU()

        # fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)   # after pooling its 64 f.maps size 8x8, 128 output numbers
        self.fc2 = nn.Linear(128, 43)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))

        x = self.pool(self.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    import torch

    model = TrafficSignCNN()

    dummy_input = torch.randn(1, 3, 32, 32)

    output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)