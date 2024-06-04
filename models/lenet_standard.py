import torch.nn.functional as F
import torch.nn as nn
import torch

class LeNet_Standard(nn.Module):
    def __init__(self, n_channels, n_classes, device, fc_in=400):
        super(LeNet_Standard, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=20, kernel_size=(5,5))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(in_features=1250, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=n_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

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