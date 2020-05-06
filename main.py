# ARTGAN neural network implementation
# Training and evaluating

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from torchvision import transforms, utils

from nn.ArtGAN import ArtGAN
from nn.Discriminator import clsNet, Enc, Discriminator
from nn.Generator import zNet, Dec, Generator
from WikiartDataset import WikiartDataset

if __name__ == '__main__':

    # We set random seed so it's reproducible
    seed = 1000
    random.seed(seed)
    torch.manual_seed(seed)

    # Local path
    path = Path().absolute()
    # TODO: import dataset and separate it in trainset and testset
    # fix constructor
    # artgan = ArtGAN(Generator(), Discriminator())
    # TODO: call training