# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import numpy as np

class ArtGAN(nn.Module):

    def __init__(self, gen, dis):
        """
        ArtGAN constructor
        :param gen: Generator
        :param dis: Discriminator
        """
        self.dis = dis
        self.gen = gen
        # TODO


    def train(self, optimizer, trainset, testset, loss, epochs=100,
              batch_size=128, img_interval=20, cuda=True):
        """
        Training function
        :param optimizer: used optimizer
        :param trainset: data for training the nn
        :param testset: data for testing the nn
        :param loss: used loss function
        :param epochs: quantity of epochs
        :param batch_size: batch size
        :param img_interval: interval to show the images produced
        by the generator
        :param cuda: if it uses gpu
        :return: Loss list, accuracy list
        """

        # We shuffle and take the batchs so that for each batch we train the Dicriminator and the Generator
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
        testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)

        # TODO: we should shuffle again at the end of each epoch?
        for epoch in range(epochs):

            # TODO: Train Discriminator


            # TODO: Train Generator


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
        pass

