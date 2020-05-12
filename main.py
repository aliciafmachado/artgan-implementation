# ARTGAN neural network implementation
# Training and evaluating

import torch
import torch.nn as nn
import numpy as np
import random
import torchvision

from pathlib import Path
from torchvision import transforms, utils

from nn.ArtGAN import ArtGAN
from WikiartDataset import WikiartDataset


def test_cifar_10():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    batch_size = 128

    print("Importing data")
    # Change root
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Creating the class")
    artgan = ArtGAN(img_size=32, num_classes=1000)

    print("Training the class")
    df = artgan.train(trainloader, testloader, epochs=5, img_interval=1, cuda=False)


if __name__ == '__main__':

    # We set random seed so it's reproducible
    seed = 1000
    random.seed(seed)
    torch.manual_seed(seed)

    # Local path
    path = Path().absolute()
    test_cifar_10()
    print("Ended!")