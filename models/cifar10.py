import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math


class FlatCifar(nn.Module):

    def __init__(self):
        super(FlatCifar, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        lr0 = 5e-4
        self.regime = lambda epoch: {
            'optimizer': 'Adam', 'lr': lr0 if epoch < 200 else lr0 * (0.95**max(epoch - 200, 0)), 'momentum': 0.5}

    def forward(self, x):
        x = self.fc1(x.view(-1, 3 * 32 * 32))
        x = self.fc2(self.relu(x)).view(-1)
        return x


def model(**kwargs):
    return FlatCifar()
