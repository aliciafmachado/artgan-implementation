# ARTGAN neural network implementation
# Generator layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class zNet(nn.Module):
    # Where dense code is transformed to latent code
    def __init__(self, input_size=100, output_size=1, output_dim=1024):
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
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv1 = nn.ConvTranspose2d(110, 1024, 4, stride=1, padding=0)

    def forward(self, x):
        out = x.view(-1, 110, 1, 1)
        out = F.relu(self.bn1(self.deconv1(out)))
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

        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d( 512, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d( 256, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d( 128, 128, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv6 = nn.ConvTranspose2d( 128,   3, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        x = torch.sigmoid(self.deconv6(x))
        return x


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