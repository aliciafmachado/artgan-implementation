# ARTGAN neural network implementation
# Training and evaluating

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms, utils

from nn.ArtGAN import ArtGAN
from nn.Discriminator import clsNet, Enc, Discriminator
from nn.Generator import zNet, Dec, Generator
from WikiartDataset import WikiartDataset

def train(net, optimizer, trainloader, testloader, loss, epochs=100,
           batch_size=128, img_interval=20, cuda=True):
    """
    Training function
    :param net: ArtGAN nn
    :param optimizer: used optimizer
    :param trainloader: data for training the nn
    :param testloader: data for testing the nn
    :param loss: used loss function
    :param epochs: quantity of epochs
    :param batch_size: batch size
    :param img_interval: interval to show the images produced
    by the generator
    :param cuda: if it uses gpu
    :return: Loss list, accuracy list
    """
    # Should the testloader look like that or like the trainloader
    # testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # TODO: Randomly intialize thetaD and thetaG
    # TODO: denote parameters of Dec, thetaDec in thetaG
    for epoch in range(epochs):
        # I think that d = 1024 but i'm not quite sure
        # take a batch
        # X_r or probably x
        # rand_sampler = torch.utils.data.RandomSampler(train_set, num_samples=batch_size, replacement=True)
        # train_sampler = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=rand_sampler)
        d = 1024
        z_hat = []

        for i in range(batch_size):
            t = 2
            # z_hat.append(m_normal(np.ones(d), np.identity(d)))

        y_hat_k = []
        # TODO: Randomly set y (guess it's an uniform distribution)

        z_y_hat = np.concatenate(z_hat, y_hat_k)  # I thought it was concatenation but not sure anymore
        # x_hat = self.gen(z_y_hat)
        # y_hat = self.dis(x_hat)

        # TODO: theta_D : backpropagation

        # z = self.dis.enc(x)  # TODO: who is x?
        # x_z = self.gen.dec(z)

        # TODO theta_G : backpropagation

        if (epoch % img_interval == 0):
            # self.show_imgs(epoch)
            a = 2


if __name__ == '__main__':

    # Local path
    path = Path().absolute()
    # TODO: import dataset and separate it in trainset and testset
    # fix constructor
    # artgan = ArtGAN(Generator(), Discriminator())
    # TODO: call training