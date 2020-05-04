# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import numpy as np

from numpy.random import multivariate_normal as m_normal


class ArtGAN(nn.Module):

    def __init__(self, gen, dis):
        """
        ArtGAN constructor
        :param gen: Generator
        :param dis: Discriminator
        """
        super(ArtGAN, self).__init__()
        self.dis = dis
        self.gen = gen
        # TODO

    # Function to save image created by our generator
    def show_imgs(self, epoch, quantity=6):
        """
        Function to produce images from our nn
        :param epoch: in which epoch we are
        :param quantity: quantity of images to be produces
        :return: images produced by the nn
        """
        # TODO

        pass
