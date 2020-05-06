# ARTGAN neural network implementation
# Generator layers

import torch
import torch.nn as nn
import numpy as np


class zNet(nn.Module):
    # Where dense code is transformed to latent code
    def __init__(self, input_size=16384, output_size=4, output_dim=1024):
        """
        zNet Constructor
        :param input_size: size of Y + Z
        :param output_size: size of the image after its transformed to latent code
        :param output_dim: size of the dimension after its transformed to latent code
        :output: Latent code 1024 x 4 x 4
        """
        super(zNet, self).__init__()
        self.output_size = output_size
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Linear(input_size, output_size**2 * output_dim),
            nn.BatchNorm2d(output_size**2 * output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, y):
        out = torch.cat([z, y], 1)
        out = self.features(out)
        out = out.view(-1, self.output_dim, self.output_size, self.output_size)
        return out


class Dec(nn.Module):
    # Latent code is upsampled via
    # deconvolution to image space
    def __init__(self, input_dim=1024):
        """
        :param input_dim: dimension of the code after zNet
        :output: Image 3 x 64 x 64
        """
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
        """
        :param znet: ZNet
        :param dec: Decoder
        """
        super(Generator, self).__init__()
        self.znet = znet
        self.dec = dec

    def forward(self, x):
        """
        :param x: Torch array of size 4096
        :return: Image 3 x 64 x 64
        """
        out = self.znet(x)
        out = self.dec(out)
        return out