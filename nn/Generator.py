# ARTGAN neural network implementation
# Generator layers

import torch
import torch.nn as nn
import numpy as np


class zNet(nn.Module):
    # Where dense code is transformed to latent code
    def __init__(self, input_size=1024, output_size=32, output_dim=1024):
        super(zNet, self).__init__()
        self.output_size = output_size
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Linear(input_size, output_size**2 * output_dim),
            nn.BatchNorm2d(output_size**2 * output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, self.output_dim, self.output_size, self.output_size)
        return out


class Dec(nn.Module):
    # Latent code is upsampled via
    # deconvolution to image space
    def __init__(self, input_dim=1024):
        super(Dec, self).__init__()

        self.features = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 1024, 4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 516, 4, stride=2, padding=1),
            nn.BatchNorm2d(516),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(516, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.features(x)


class Generator(nn.Module):
    # G layers
    def __init__(self, znet, dec):
        super(Generator, self).__init__()
        self.znet = znet
        self.dec = dec

    def forward(self, x):
        out = self.znet(x)
        out = self.dec(out)
        return out