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


    def train(self, optimizer, train_set, test_set, loss, epochs=100,
              batch_size=128, img_interval=20, cuda=True):
        """
        Training function
        :param optimizer: used optimizer
        :param train_set: data for training the nn
        :param test_set: data for testing the nn
        :param loss: used loss function
        :param epochs: quantity of epochs
        :param batch_size: batch size
        :param img_interval: interval to show the images produced
        by the generator
        :param cuda: if it uses gpu
        :return: Loss list, accuracy list
        """

        # Should the testloader look like that or like the trainloader
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

        # TODO: Randomly intialize thetaD and thetaG
        # TODO: denote parameters of Dec, thetaDec in thetaG
        for epoch in range(epochs):
            # I think that d = 1024 but i'm not quite sure
            # take a batch
            # X_r or probably x
            rand_sampler = torch.utils.data.RandomSampler(train_set, num_samples=batch_size, replacement=True)
            train_sampler = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=rand_sampler)
            d = 1024
            z_hat = []

            for i in range(batch_size):
                z_hat.append(m_normal(np.ones(d), np.identity(d)))

            y_hat_k = []
            # TODO: Randomly set y (guess it's an uniform distribution)

            z_y_hat = np.concatenate(z_hat, y_hat_k) # I thought it was concatenation but not sure anymore
            x_hat = self.gen(z_y_hat)
            y_hat = self.dis(x_hat)

            # TODO: theta_D : backpropagation

            z = self.dis.enc(x) # TODO: who is x?
            x_z = self.gen.dec(z)

            # TODO theta_G : backpropagation

            if(epoch % img_interval == 0):
                self.show_imgs(epoch)


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

