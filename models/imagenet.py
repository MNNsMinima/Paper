import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
from torch.utils.serialization import load_lua

mean = load_lua('/home/ehoffer/Torch/FLAT/mean_tinyImageNet.t7').float()
w1 = load_lua('/home/ehoffer/GoogleDrive/PyTorch/flatnet/w1_cov_1.t7')
b1 = load_lua('/home/ehoffer/GoogleDrive/PyTorch/flatnet/b1.t7')
w2 = load_lua('/home/ehoffer/GoogleDrive/PyTorch/flatnet/w2.t7')
b2 = load_lua('/home/ehoffer/GoogleDrive/PyTorch/flatnet/b2.t7')

def center(x):
    return 255*x - mean

class FlatImageNet(nn.Module):

    def __init__(self):
        super(FlatImageNet, self).__init__()
        self.fc1 = nn.Linear(3*64*64, 105)
        self.fc1.weight.data.copy_(w1)
        self.fc1.bias.data.copy_(b1)


        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(105, 1)
        self.fc2.weight.data.copy_(w2)
        self.fc2.bias.data.copy_(b2)



        self.regime = lambda epoch: {
            'optimizer': 'Adam', 'lr': 1e-6, 'weight_decay': 0}
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Lambda(center)
        t = transforms.Compose([
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ])
        self.input_transform = {
            'train': t, 'eval': t
        }


    def forward(self, x):
        x = self.fc1(x.view(-1,3*64*64))
        x = self.fc2(self.relu(x)).view(-1)
        return x


def model(**kwargs):
    return FlatImageNet()
