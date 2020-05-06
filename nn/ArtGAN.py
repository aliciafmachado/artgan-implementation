# ARTGAN neural network implementation
# The neural network itself

import torch
import torch.nn as nn
import utils
from tqdm import tqdm

from nn.Discriminator import clsNet, Enc, Discriminator
from nn.Generator import zNet, Dec, Generator


class ArtGAN:

    def __init__(self,
                 img_size=64,
                 input_dim_enc=3,
                 z_dim=None,
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
        super(ArtGAN, self).__init__()
        # Inputs
        self.img_size = img_size
        self.input_dim_enc = input_dim_enc # usually 3
        self.num_classes = num_classes
        self.out_dim_zNet = out_dim_zNet
        self.out_size_zNet = self.img_size/2**4
        self.z_dim = z_dim
        self.out_size_enc = self.out_dim_zNet * self.out_size_zNet ** 2
        if z_dim is None:
            self.z_dim = self.out_size_enc - self.num_classes

        # Nets
        self.znet = zNet(input_size=self.z_dim + self.num_classes,
                         output_size=self.out_size_zNet, output_dim=self.out_dim_zNet)
        self.clsnet = clsNet(input_size=self.out_size_enc, num_classes=self.num_classes)
        self.enc = Enc(input_dim=self.input_dim_enc, output_size=self.out_size_enc)
        self.dec = Dec(input_dim=self.out_dim_zNet)
        self.D = Discriminator(self.clsnet, self.enc)
        self.G = Generator(self.znet, self.dec)

        # Optimisers
        self.opt_D = torch.optim.Adadelta(self.D.parameters(), lr=0.001, rho=0.9, eps=1e-06, weight_decay=0)
        self.opt_G = torch.optim.Adadelta(self.G.parameters(), lr=0.001, rho=0.9, eps=1e-06, weight_decay=0)

        # Loss functions
        self.BCE_loss = nn.BCELoss()
        self.MSE_loss = nn.MSELoss()

    def show_imgs(self, epoch, fixed_noise=None):
        """
        Function to produce images from our nn
        :param epoch: in which epoch we are
        :param quantity: quantity of images to be produces
        :return: images produced by the nn
        """
        # TODO
        pass

    def train(self, trainloader, testloader, epochs=10,
              img_interval=20, batch_size=128, cuda=True):
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

        # To evaluate how our net is learning
        fixed_noise = torch.randn(64, self.z_dim, 1, 1)

        for epoch in range(epochs):

            # import tqdm
            for i, data in enumerate(tqdm(trainloader), 0):
                # zero grad
                self.opt_D.zero_grad()
                self.opt_G.zero_grad()
                # get the inputs
                x_r, k = data
                # generate z_hat
                z_hat = utils.gen_z(batch_size, self.z_dim)
                # generate Y_k
                y_k = utils.gen_yk(batch_size, self.num_classes)
                if cuda:
                    x_r = x_r.type(torch.cuda.FloatTensor)
                    k = k.type(torch.cuda.LongTensor)
                    z_hat = z_hat.type(torch.cuda.FloatTensor)
                    y_k = y_k.type(torch.cuda.LongTensor)

                # calculate Y
                y = self.D(x_r)
                # calculate X_hat - I'm not quite sure if it's like that
                x_hat = self.G(z_hat, y_k)
                # Calculate Y_hat
                y_hat = self.D(x_hat)
                # Fake Y
                y_fake = utils.fake_v(batch_size, self.num_classes)
                # update D
                d_real_loss = self.BCE_loss(y, k)
                d_fake_loss = self.BCE_loss(y_hat, y_fake)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()

                self.opt_D.step()

                # calculate Z
                z = self.enc(x_r)
                # calculate X_hat_z
                x_hat_z = self.dec(z)
                # update G
                g_loss_adv = self.BCE_loss(y_hat, y_k)
                g_loss_l2 = self.MSE_loss(x_hat_z, x_r)
                g_loss = g_loss_adv + g_loss_l2
                g_loss.backward()

                self.opt_G.step()

            # TODO: Calculate and show loss and accuracy values as well as put them
            # TODO: in a list to be shown
            # TODO: how to use the testloader? I think that it may be see if dis can
            # TODO: know how is fake and how is not...