import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms, utils


class WikiartDataset(Dataset):

    def __init__(self, type, file, dir, mode, transform):
        """
        :param type: 0: Style, 1: Genre, 2: Artist
        :param file: .csv file that contains the path and classification
        :param dir: directory of the image files
        :param mode: 'Train' or 'Test'
        :param transform: image transformation
        """
        self.type = type
        self.dir = dir
        self.df = pd.read_csv(os.path.join(self.dir, file), header=None, names=["file", "label"])
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __str__(self):
        text = "Type: "
        if self.type == 0:
            text += "Style"
        elif self.type == 1:
            text += "Genre"
        elif self.type == 2:
            text += "Artist"
        text += ", mode: "
        text += self.mode
        text += ", number of files: "
        text += str(len(self.df))
        return text

    def __getitem__(self, idx):
        file = self.df.loc[idx, "file"]
        label = self.df.loc[idx, "label"]
        img = Image.open(os.path.join(self.dir, file))
        img = self.transform(img)
        return img, label