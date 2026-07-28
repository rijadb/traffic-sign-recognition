import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TrafficSignResNet(nn.Module):

    def __init__(self, pretrained=False):

        super().__init__()

        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)        # transfer learning, learnec weights
        else:
            self.model = resnet18(weights=None)         # create network (start with random weights)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,              # return number of inputs in last layer
            43                                      # we need 43 for our dataset, not 1000
        )

    def forward(self, x):

        return self.model(x)