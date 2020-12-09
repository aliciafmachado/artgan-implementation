# ARTGAN neural network implementation
# File that plots the loss and produces a grid with images created

import torch
import random
import torchvision
import os
import argparse
import utils as ut
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from pathlib import Path
from torchvision import transforms, utils

from nn.ArtGAN import ArtGAN
from nn.Generator import Generator, Dec, zNet
from nn.Discriminator import Discriminator, Enc, clsNet
from WikiartDataset import WikiartDataset
from mpl_toolkits.axes_grid1 import ImageGrid


def main():
    # Plot loss graphs
    # Create grids with images:
    # From cifar-10 and artgan

    # Parser
    # Here you can choose which class dataset are you working with
    # You also can choose the version you want to put in the name of the file
    # You can choose which net to use
    # And you say if you are using a saved net
    # You also say if you want to produce a grid
    parser = argparse.ArgumentParser()
    parser.add_argument("class_dataset", type=str)
    parser.add_argument("version", type=int)
    parser.add_argument("--net", type=str)
    parser.add_argument("--save", type=int, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--grid", type=str, default=None)
    args = parser.parse_args()

    # Training using wikiart dataset
    class_dataset = args.class_dataset  # style - artist - genre
    version = args.version  # number of the version

    num_folder = "../save/" + class_dataset + "_test_v" + str(version)
    # Check if you are in the folder Deep_Learning_Dataset
    if not os.path.exists(num_folder):
        os.makedirs(num_folder)

    # Read classifications in txt
    if class_dataset == "cifar-10":
        with open('../cifar-10/' + class_dataset + '_class.txt', 'r') as f:
            cl = [line.strip() for line in f]

        cl.append("fake")
        classes = tuple(cl)
        n_classes = len(classes) - 1

    else:
        with open('../wikiart/' + class_dataset + '_class.txt', 'r') as f:
            cl = [line.strip() for line in f]

        cl.append("fake")
        classes = tuple(cl)
        n_classes = len(classes) - 1

    if args.loss:
        data = pd.read_csv(args.loss)
        save_loss_graph(data, class_dataset, num_folder)

    if args.save:
        use_cuda = True

        if use_cuda and torch.cuda.is_available():
            checkpoint = torch.load(args.net)

        else:
            checkpoint = torch.load(args.net, map_location=torch.device("cpu"))
        gen = Generator(zNet(input_size=100 + n_classes), Dec())
        dis = Discriminator(clsNet(num_classes=n_classes), Enc())

        if use_cuda and torch.cuda.is_available():
            gen.cuda()
            dis.cuda()

        gen.load_state_dict(checkpoint["G"])
        dis.load_state_dict(checkpoint["D"])
        save_imgs(gen, dis, classes, n_classes, num_folder)

    if args.grid:
        save_grid(class_dataset, args.grid)


# Here we save the training and test loss in an image
def save_loss_graph(data, class_dataset, path):
    data_ep = data.groupby(['epoch']).mean()
    data_ep.rename(columns={"d_loss": "Discriminator", "g_loss": "Generator"}, inplace=True)
    data_ep.info()

    plt.figure()
    ax_d = data_ep["Discriminator"].plot(x="epoch", y="Discriminator", color="blue", legend=True)
    ax_d.set_xlabel("Epoch")
    ax_d.set_ylabel("loss")

    ax_g = data_ep["Generator"].plot(x="epoch", y="Generator", color="red", legend=True)
    ax_g.set_xlabel("Epoch")
    ax_g.set_ylabel("Loss")
    plt.title("Loss of " + class_dataset)
    plt.savefig(path + "/" + "loss.png")


# Here we save the image using save_img from utils
def save_imgs(gen, dis, classes, n_classes, num_folder):
    ut.save_img(gen, dis, 0, classes, test_num=n_classes, path=num_folder)


# Here we save the grid with images produced by the Generator
def save_grid(class_dataset, path):
    imgs = []
    rg = 3

    if class_dataset == "cifar-10":
        rg = 5

    for i in range(rg):
        name = path + "/" + class_dataset + "_fake_" + str(i) + ".jpg"
        img = mpimg.imread(name)
        imgs.append(img)

    for i in range(rg):
        name = path + "/" + class_dataset + "_not_fake_" + str(i) + ".jpg"
        img = mpimg.imread(name)
        imgs.append(img)

    # Now print the grid
    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, rg),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.axis('off')
        ax.imshow(im)

    plt.savefig(path + "/" + class_dataset + "_grid.png")


if __name__ == '__main__':
    main()