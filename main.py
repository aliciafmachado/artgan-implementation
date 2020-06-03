# ARTGAN neural network implementation
# Training and evaluating

import torch
import random
import torchvision
import os
import argparse

from pathlib import Path
from torchvision import transforms, utils

from nn.ArtGAN import ArtGAN
from nn.Generator import Generator, Dec, zNet
from nn.Discriminator import Discriminator, Enc, clsNet
from WikiartDataset import WikiartDataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # We set random seed so it's reproducible
    seed = 1000
    random.seed(seed)
    torch.manual_seed(seed)

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("class_dataset", type=str)
    parser.add_argument("version", type=int)
    parser.add_argument("--retrain", type=str, default=None)
    args = parser.parse_args()

    # Training using wikiart dataset
    class_dataset = args.class_dataset  # style - artist - genre
    version = args.version  # number of the version

    num_folder = "../save/" + class_dataset + "_v" + str(version)
    # Check if you are in the folder Deep_Learning_Dataset
    if not os.path.exists(num_folder):
        os.makedirs(num_folder)

    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
        ])

    # Choice of kind of dataset
    print("Creating dataset with wikiart")
    trainset_wikiart = WikiartDataset(0, class_dataset + "_train.csv", "../wikiart/", 'Train', transform)
    testset_wikiart = WikiartDataset(0, class_dataset + "_val.csv", "../wikiart/", 'Test', transform)
    with open('../wikiart/' + class_dataset + '_class.txt', 'r') as f:
        cl = [line.strip() for line in f]
    cl.append("fake")
    classes = tuple(cl)
    n_classes = len(classes) - 1

    batch_size = 128
    print("Calling loader")
    trainloader_wikiart = torch.utils.data.DataLoader(trainset_wikiart, batch_size=batch_size, shuffle=True)
    testloader_wikiart = torch.utils.data.DataLoader(testset_wikiart, batch_size=batch_size, shuffle=True)

    if args.retrain:
        checkpoint = torch.load(args.retrain)
        gen = Generator(zNet(input_size=100 + n_classes), Dec())
        dis = Discriminator(clsNet(num_classes=n_classes), Enc())
        g_op = torch.optim.RMSprop(gen.parameters(), lr=0.001, alpha=0.9)
        d_op = torch.optim.RMSprop(dis.parameters(), lr=0.001, alpha=0.9)
        epo = checkpoint['epoch']
        d_op = utils.exp_lr_scheduler(d_op, epo)
        g_op = utils.exp_lr_scheduler(g_op, epo)
        gen.load_state_dict(checkpoint["G"])
        dis.load_state_dict(checkpoint["D"])
        d_op.load_state_dict(checkpoint["opt_D"])
        g_op.load_state_dict(checkpoint["opt_G"])
        net = ArtGAN(img_size=64, input_dim_enc=3,
                     z_dim=100, num_classes=n_classes,
                     out_dim_zNet=1024, G=gen, D=dis, retrain=True)

    else:
        net = ArtGAN(img_size=64, input_dim_enc=3,
                     z_dim=100, num_classes=n_classes,
                     out_dim_zNet=1024)

    use_cuda = True

    if use_cuda and torch.cuda.is_available():
        print("using cuda")
        net.cuda()

    print("Beginning training . . .")
    if args.retrain:
        d_loss_l, g_loss_l = net.train(trainloader_wikiart, None, classes, epochs=100, batch_size=batch_size,
                                       cuda=use_cuda and torch.cuda.is_available(), path=num_folder, g_op=g_op,
                                       d_op=d_op, init_epoch=epo + 1)
    else:
        d_loss_l, g_loss_l = net.train(trainloader_wikiart, None, classes, epochs=100, batch_size=batch_size,
                                       cuda=use_cuda and torch.cuda.is_available(), path=num_folder)

    print("Ended!")

if __name__ == '__main__':
    main()