# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import numpy as np

class ArtGAN(nn.Module):

    def __init__(self, gen, dis):
        self.dis = dis
        self.gen = gen
        # TODO

    def forward(self):
        pass
        # does it make sense ?

    def train(self):
        pass
        # TODO


