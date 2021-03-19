"""
Dispose of drug data, gene expression data and drug response data
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch

def DefaultLoader(piece):

    piece = list(piece)

    dLatentVec = piece[3: 59]
    dLatentVec = np.array(dLatentVec)
    dLatentVec = torch.from_numpy(dLatentVec).float()

    # Encoded gene expression data is 256-dimensional.
    # When used to dispose of raw gene expression data (filtered by CGC),
    # "315" should be set as "656"
    geLatentVec = piece[59: 315]
    geLatentVec = np.array(geLatentVec)
    geLatentVec = torch.from_numpy(geLatentVec).float()

    target = piece[2:3]
    target = np.array(target)
    target = torch.from_numpy(target).float()

    return geLatentVec, dLatentVec, target


class LatentDataset(Dataset):
    def __init__(self, data, train0val1test2, loader=DefaultLoader):
        self.data = data
        self.loader = loader
        self.train0val1test2 = train0val1test2

    def __getitem__(self, index):
        if self.train0val1test2 == 0:
            piece = self.data.loc[index]
        elif self.train0val1test2 == 1:
            # For pan cancer dataset
            # "3000" should be set as "12240"
            piece = self.data.loc[index+12240]
        else:
            # For pan cancer dataset
            # "3165" should be set as "12920"
            piece = self.data.loc[index + 12920]
        geLatentVec, dLatentVec, target = self.loader(piece)
        return geLatentVec, dLatentVec, target

    def __len__(self):
        return len(self.data)
