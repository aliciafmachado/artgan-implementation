# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import numpy as np
import math
import random
# TODO: Import tqdm

from nn.Discriminator import clsNet, Enc, Discriminator
from nn.Generator import zNet, Dec, Generator


class ArtGAN:

    def __init__(self, img_size=64, input_dim=3, z_dim=1024, num_classes=10):
        """
        ArtGAN constructor
        :param img_size:
        :param input_dim:
        :param z_dim:
        :param num_classes:
        """
        super(ArtGAN, self).__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        znet = zNet(z_dim + num_classes, output_size=4, output_dim=1024)
        clsnet = clsNet(input_size=1024, num_classes=num_classes)
        enc = Enc(input_dim=input_dim, size_output=z_dim)
        dec = Dec(input_dim=input_dim)
        self.dis = Discriminator(clsnet, enc)
        self.gen = Generator(znet, dec)

    def show_imgs(self, epoch, fixed_noise=None):
        """
        Function to produce images from our nn
        :param epoch: in which epoch we are
        :param quantity: quantity of images to be produces
        :return: images produced by the nn
        """
        # TODO
        pass

    @staticmethod
    def gen_z(batch_size, d):
        """
        Produces a batch of random normal vectors
        :param batch_size: batch_size
        :param d: size of the normal vector
        :return: batch of random normal vectors
        """
        return torch.rand(batch_size, d)

    @staticmethod
    def gen_yk(batch_size, num_classes):
        """

        :param batch_size:
        :param num_classes:
        :return:
        """
        v = torch.zeros(batch_size, num_classes)
        for row in v:
            num = random.randint(0, num_classes - 1)
            row[num] = 1
        return v

    @staticmethod
    def fake_v(batch_size, num_classes):
        """

        :param batch_size:
        :param num_classes:
        :return:
        """
        v = torch.zeros(batch_size, num_classes + 1)
        v[:, -1] = 1
        return v

    def train(self, optimizers, criterions, trainloader, testloader, epochs=100,
              batch_size=128, img_interval=20, cuda=True):
        """
        Training function
        :param optimizers: used optimizers (2 entries)
        :param criterions: loss functions (3 entries)
        :param trainloader: dataset in batch
        :param testloader: testset in batch
        :param epochs: epochs
        :param batch_size: batch_size
        :param img_interval: interval for showing the images produced
        :param cuda: usage of gpu
        :return: Loss list, accuracy list
        """

        # Adapt utilisation of GPU
        # TODO: There are two other optimizations in the paper but not sure what they mean
        opt_d, opt_g = optimizers

        # The 3 different loss functions used
        f_d_loss, f_adv_loss, f_l2_loss = criterions

        # To evaluate how our net is learning
        fixed_noise = torch.randn(64, self.z_dim, 1, 1)

        loss_train = []
        loss_test = []
        total = 0

        for epoch in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            # import tqdm
            for i, data in enumerate(tqdm(trainloader), 0):
                # zero grad
                opt_d.zero_grad()
                opt_g.zero_grad()
                # get the inputs
                x_r, k = data
                # generate z_hat
                z_hat = self.gen_z(batch_size, self.z_dim)
                # generate Y_k
                y_k = self.gen_yk(batch_size, self.num_classes)
                if cuda:
                    x_r = x_r.type(torch.cuda.FloatTensor)
                    k = k.type(torch.cuda.LongTensor)
                    z_hat = z_hat.type(torch.cuda.FloatTensor)
                    y_k = y_k.type(torch.cuda.LongTensor)

                # calculate Y
                y = self.dis(x_r)
                # calculate X_hat - I'm not quite sure if it's like that
                x_hat = self.gen(z_hat, y_k)
                # Calculate Y_hat
                y_hat = self.dis(x_hat)
                # Fake Y
                y_fake = self.fake_v(batch_size, self.num_classes)
                # update D
                d_real_loss = f_d_loss(y, k)
                d_fake_loss = f_d_loss(y_hat, y_fake)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()

                opt_d.step()

                # calculate Z - not sure if it's the same x_r but hope so
                z = self.enc(x_r)
                # calculate X_hat_z
                x_hat_z = self.gen.dec(z)
                # update G
                g_loss_adv = f_adv_loss(y_hat, y_k)
                g_loss_l2 = f_l2_loss(x_hat_z, x_r)
                g_loss = g_loss_adv + g_loss_l2
                g_loss.backward()

                opt_g.step()

            if epoch % img_interval == 0:
                self.show_imgs(epoch, fixed_noise=fixed_noise)

            # TODO: Calculate and show loss and accuracy values as well as put them
            # TODO: in a list to be shown
            # TODO: how to use the testloader? I think that it may be see if dis can
            # TODO: know how is fake and how is not...