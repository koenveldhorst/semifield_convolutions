import sys
sys.path.append('../')

import torch.nn.functional as F
import torch.nn as nn
import torch

from semifield_integration.semi_pooling import SemiPool2dLearnable

class LeNet_Learnable(nn.Module):
    def __init__(self, n_channels, n_classes, device, ks=3, fc_in=1250, initial_scale=1.0):
        super(LeNet_Learnable, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=20, kernel_size=(5,5))
        self.relu1 = nn.ReLU()
        semifield1 = (self._maxvalues, torch.add, -1.0 * torch.inf, 0)
        self.pool1 = SemiPool2dLearnable(semifield=semifield1, c_in=20, c_out=20, kernel_size=ks, stride=2, device=device, padding='same')

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        semifield2 = (self._maxvalues, torch.add, -1.0 * torch.inf, 0)
        self.pool2 = SemiPool2dLearnable(semifield=semifield2, c_in=50, c_out=50, kernel_size=ks, stride=2, device=device, padding='same')

        self.fc1 = nn.Linear(in_features=fc_in, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=n_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _maxvalues(self, a, dim=1):
        return torch.max(a, dim=dim).values

    def _minvalues(self, a, dim=1):
        return torch.min(a, dim=dim).values

    def _calculate_num_features(self, x):
        with torch.no_grad():
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        output = self.log_softmax(self.fc2(x))
        return output