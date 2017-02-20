import os
import torch
import torchnet.dataset as dataset
import torchvision.datasets as datasets
from torch.utils.serialization import load_lua
# os.path.expanduser('~/Datasets')
__DATASETS_DEFAULT_PATH = '/media/SSD/Datasets/'


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if name == 'cifar10_whitened':
        x = load_lua('/home/ehoffer/Datasets/Cifar10/cifar10_whitened.t7')
        if train:
            return dataset.TensorDataset(
                [x['trainData']['data'],  (x['trainData']['labels']-5.5).sign()])
        else:
            return dataset.TensorDataset(
                [x['testData']['data'],  (x['testData']['labels']-5.5).sign()])
    if name == 'tinyImagenet':
        if train:
            return datasets.ImageFolder(root= '/home/ehoffer/Datasets/ImageNet/tiny',
                                        transform=transform,
                                        target_transform=target_transform)
            # x = load_lua('/home/ehoffer/Datasets/ImageNet/tinyImageNet.t7')
            # return dataset.TensorDataset(
            #     [x['data'],  (x['label'].float()-500.5).sign()])
        else:
            return dataset.TensorDataset(
                [torch.rand(100,3,64,64).float(),  torch.rand(100).float()])
    elif name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
