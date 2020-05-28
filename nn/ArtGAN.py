# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
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
                 G=None,
                 D=None):
        """
        ArtGAN constructor
        :param img_size:
        :param input_dim:
        :param z_dim:
        :param num_classes:
        """
        if G is None and D is None:
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
            self.znet = zNet(input_size=self.z_dim + self.num_classes)
            self.clsnet = clsNet(num_classes=self.num_classes)
            self.enc = Enc()
            self.dec = Dec()
            self.D = Discriminator(self.clsnet, self.enc)
            self.G = Generator(self.znet, self.dec)

        else:
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
            self.G = G
            self.D = D

    def cuda(self):

        self.D.cuda()
        self.G.cuda()

    def train(self, trainloader, testloader, classes, epochs=10,
              img_interval=1, batch_size=64, cuda=True, path=None):
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

        pd_loss = pd.DataFrame(columns=['epoch', 'd_loss', 'g_loss'])
        path_loss_folder = path + "/Wikiart_loss"
        path_loss = path_loss_folder + "/loss.csv"
        if not os.path.exists(path_loss_folder):
            os.makedirs(path_loss_folder)
        pd_loss.to_csv(path_loss, index=False)

        utils.save_img(self.G, self.D, 1, classes, path=path, test_num=len(classes))

        for epoch in range(epochs):
            # Save loss
            g_loss_l = []
            d_loss_l = []

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

                t_zeros = torch.zeros(b_s, 1)
                k_hot = F.one_hot(k, self.num_classes + 1)
                # print("k_hot = ", k_hot.size())
                # This other cuda is so that y_k_hot is created correctly

                y_fake = y_fake.type(torch.int64)
                y_k_hot = F.one_hot(y_fake, self.num_classes + 1)

                if cuda:
                    y_k = y_k.type(torch.cuda.FloatTensor)
                    x_r = x_r.type(torch.cuda.FloatTensor)
                    k = k.type(torch.cuda.LongTensor)
                    z_hat = z_hat.type(torch.cuda.FloatTensor)
                    y_k = y_k.type(torch.cuda.FloatTensor)
                    k_hot = k_hot.type(torch.cuda.FloatTensor)
                    y_k_hot = y_k_hot.type(torch.cuda.FloatTensor)
                    t_zeros = t_zeros.type(torch.cuda.FloatTensor)

                else:
                    k_hot = k_hot.type(torch.FloatTensor)
                    y_k_hot = y_k_hot.type(torch.FloatTensor)

                # calculate X_hat
                in_G = torch.cat([z_hat, y_k], 1)
                # print("y_k_hot = ", y_k_hot.size())
                # calculate Y
                y = self.D(x_r)
                # Calculate Y_hat
                x_hat = self.G(in_G)
                y_hat = self.D(x_hat)
                # update D
                d_real_loss = F.binary_cross_entropy(y, k_hot)
                d_fake_loss = F.binary_cross_entropy(y_hat, y_k_hot)
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
                    [y_k, t_zeros], 1)
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

            d = {'epoch': epoch, 'd_loss': d_loss_l, 'g_loss': g_loss_l}
            pd_loss = pd.read_csv(path_loss)
            pd_loss = pd_loss.append(pd.DataFrame(data=d), ignore_index=True)
            pd_loss.to_csv(path_loss, index=False)

            # print image
            if ((epoch + 1) % img_interval == 0):
                utils.save_img(self.G, self.D, epoch, classes,path=path, test_num=len(classes))
                name_net_folder = path + "/Wikiart_nets"
                name_net = name_net_folder + "/nn_" + str(epoch) + ".pt"
                if not os.path.exists(name_net_folder):
                    os.makedirs(name_net_folder)
                torch.save({'epoch': epoch,
                            'G': self.G.state_dict(),
                            'D': self.D.state_dict(),
                            'opt_G': g_opt.state_dict(),
                            'opt_D': d_opt.state_dict(),
                            }, name_net)

        return d_loss_l, g_loss_l

