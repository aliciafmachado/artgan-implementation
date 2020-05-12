# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from nn.Discriminator import clsNet, Enc, Discriminator
from nn.Generator import zNet, Dec, Generator


class ArtGAN:

    def __init__(self,
                 img_size=64,
                 input_dim_enc=3,
                 z_dim=100,
                 num_classes=10,
                 out_dim_zNet=1024,
                 ):
        """
        ArtGAN constructor
        :param img_size:
        :param input_dim:
        :param z_dim:
        :param num_classes:
        """
        # Inputs
        self.img_size = img_size
        self.input_dim_enc = input_dim_enc  # usually 3
        self.num_classes = num_classes
        self.out_dim_zNet = out_dim_zNet
        self.out_size_zNet = int(self.img_size / 2 ** 4)
        self.z_dim = z_dim
        self.out_size_enc = int(self.out_dim_zNet * self.out_size_zNet ** 2)
        if z_dim is None:
            self.z_dim = self.out_size_enc - self.num_classes
        # Nets
        self.znet = zNet(input_size=self.z_dim + self.num_classes,
                         output_size=self.out_size_zNet, output_dim=self.out_dim_zNet)
        self.clsnet = clsNet(input_size=self.out_size_enc, num_classes=self.num_classes)
        self.enc = Enc(input_dim=self.input_dim_enc)
        self.dec = Dec(input_dim=self.out_dim_zNet)
        self.D = Discriminator(self.clsnet, self.enc)
        self.G = Generator(self.znet, self.dec)

        # Loss functions
        self.MSE_loss = nn.MSELoss()

    def cuda(self):

        self.D.cuda()
        self.G.cuda()

    def train(self, trainloader, testloader, epochs=10,
              img_interval=1, batch_size=64, cuda=True):
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

        lr_init = 0.001

        g_opt = torch.optim.RMSprop(self.G.parameters(), lr=lr_init, alpha=0.9)
        d_opt = torch.optim.RMSprop(self.D.parameters(), lr=lr_init, alpha=0.9)

        # To evaluate how our net is learning
        fixed_noise = utils.gen_z(1, self.z_dim)
        y_k_fixed = utils.gen_yk(1, self.num_classes)

        g_loss_l = []
        d_loss_l = []
        for epoch in range(epochs):

            # Decay in the learning rate
            d_opt = utils.exp_lr_scheduler(d_opt, epoch)
            g_opt = utils.exp_lr_scheduler(g_opt, epoch)

            # import tqdm
            for i, data in enumerate(tqdm(trainloader), 0):

                # zero grad
                d_opt.zero_grad()
                # get the inputs
                x_r, k = data
                b_s = len(k)
                # generate z_hat
                z_hat = utils.gen_z(b_s, self.z_dim)
                # print("z_hat = ", z_hat.size())
                # generate Y_k and its label
                y_k = utils.gen_yk(b_s, self.num_classes)
                # print("y_k = ", y_k.size())
                # gen fakes
                y_fake = utils.fake_v(b_s, self.num_classes)
                # print("y_fake = ", y_fake.size())
                if cuda:
                    y_k = y_k.type(torch.cuda.FloatTensor)
                    x_r = x_r.type(torch.cuda.FloatTensor)
                    k = k.type(torch.cuda.LongTensor)
                    z_hat = z_hat.type(torch.cuda.FloatTensor)
                    y_k = y_k.type(torch.cuda.FloatTensor)
                # calculate X_hat
                in_G = torch.cat([z_hat, y_k.type(torch.cuda.FloatTensor)], 1)
                k_hot = F.one_hot(k, self.num_classes + 1)
                # print("k_hot = ", k_hot.size())
                y_k_hot = F.one_hot(y_fake.type(torch.int64), self.num_classes + 1)
                # print("y_k_hot = ", y_k_hot.size())
                # calculate Y
                y = self.D(x_r)
                # Calculate Y_hat
                x_hat = self.G(in_G)
                y_hat = self.D(x_hat)
                # update D
                d_real_loss = F.binary_cross_entropy(y, k_hot.type(torch.cuda.FloatTensor))
                d_fake_loss = F.binary_cross_entropy(y_hat, y_k_hot.type(torch.cuda.FloatTensor))
                d_loss = d_real_loss + d_fake_loss
                d_loss_l.append(d_loss.item())
                d_loss.backward(retain_graph=True)
                d_opt.step()

                # zero grad
                g_opt.zero_grad()

                # adversarial loss
                new_y_hat = self.D(x_hat)
                # print("new_y_hat = ", new_y_hat.size())
                new_y_k_hot = torch.cat(
                    [y_k.type(torch.cuda.FloatTensor), torch.zeros(b_s, 1).type(torch.cuda.FloatTensor)], 1)
                # print("new_y_k_hot = ", new_y_k_hot.size())
                g_loss_adv = F.binary_cross_entropy(new_y_hat, new_y_k_hot)

                # L2 loss
                # calculate z
                z = self.D.enc(x_r)
                # print("z = ", z.size())
                # calculate X_hat_z
                x_hat_z = self.G.dec(z)
                g_loss_l2 = torch.mean((x_hat_z - x_r) ** 2)

                # + g_loss_adv
                g_loss = g_loss_l2 + g_loss_adv
                g_loss.backward()
                g_loss_l.append(g_loss.item())
                g_opt.step()

        return d_loss_l, g_loss_l

