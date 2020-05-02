# ARTGAN neural network implementation
# Discriminator layers

import torch
import torch.nn as nn
import numpy as np


class clsNet(nn.Module):
    # Latent features is used for classification
    def __init__(self, input_size=1024, num_classes=10):
        super(clsNet, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, num_classes),
            nn.BatchNorm2d(num_classes),
            nn.ReLU,
        )

    def forward(self, x):
        return self.features(x)


class Enc(nn.Module):
    # Image is enconded via convolution
    # to latent features
    def __init__(self, input_dim=1024, num_output=10):
        super(Enc, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(1024, num_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, clsnet, enc):
        super(Discriminator, self).__init__()
        self.clsnet = clsnet
        self.enc = enc

    def forward(self, x):
        out = self.clsnet(x)
        out = self.enc(out)
        return out