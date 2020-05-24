# ARTGAN neural network implementation
# Training and evaluating

import torch
import torch.nn as nn
import numpy as np
import random
import torchvision
import os

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
    df = artgan.train(trainloader, testloader, epochs=100, img_interval=1, cuda=False)

def main():
    # We set random seed so it's reproducible
    seed = 1000
    random.seed(seed)
    torch.manual_seed(seed)

    # Local path
    path = Path().absolute()

    # Training using wikiart dataset
    class_dataset = "style"  # style - artist - genre
    version = 1  # number of the version
    num_classes = 27  # style: 27 - artist: 23 - genre: 10

    num_folder = class_dataset + "_v" + str(version)
    # Check if you are in the folder Deep_Learning_Dataset
    if not os.path.exists(num_folder):
        os.makedirs(num_folder)

    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
        ])

    # Choice of kind of dataset
    trainset_wikiart = WikiartDataset(0, class_dataset + "_train.csv", "../wikiart/", 'Train', transform)
    testset_wikiart = WikiartDataset(0, class_dataset + "_val.csv", "../wikiart/", 'Test', transform)
    with open('../wikiart/' + class_dataset + '_class.txt', 'r') as f:
        classes = tuple([line.strip() for line in f])
    n_classes = len(classes)

    batch_size = 1
    trainloader_wikiart = torch.utils.data.DataLoader(trainset_wikiart, batch_size=batch_size, shuffle=True)
    testloader_wikiart = torch.utils.data.DataLoader(testset_wikiart, batch_size=batch_size, shuffle=True)

    net = ArtGAN(img_size=64, input_dim_enc=3,
                 z_dim=100, num_classes=n_classes,
                 out_dim_zNet=1024)

    use_cuda = True

    if use_cuda and torch.cuda.is_available():
        print("using cuda")
        net.cuda()

    d_loss_l, g_loss_l = net.train(trainloader_wikiart, None, epochs=1)

    print("Ended!")

if __name__ == '__main__':
    main()