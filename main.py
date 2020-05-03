# ARTGAN neural network implementation
# Training and evaluating

import torch
import torch.nn as nn
import numpy as np

from nn.ArtGAN import ArtGAN
from nn.Discriminator import clsNet, Enc, Discriminator
from nn.Generator import zNet, Dec, Generator

# make imports of dataset
# create class and call training

if __name__ == '__main__':
    # fix constructor
    artgan = ArtGAN(Generator(), Discriminator())
    # call training
    #TODO