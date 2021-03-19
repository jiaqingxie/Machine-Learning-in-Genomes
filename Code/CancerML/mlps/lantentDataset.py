from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch

def DefaultLoader(piece):
    piece = list(piece)

    dLatentVec = piece[3: 63]
    dLatentVec = np.array(dLatentVec)
    dLatentVec = torch.from_numpy(dLatentVec).float()

    geLatentVec = piece[63: 319]
    geLatentVec = np.array(geLatentVec)
    geLatentVec = torch.from_numpy(geLatentVec).float()

    target = piece[2:3]
    target = np.array(target)
    target = torch.from_numpy(target).float()

    return geLatentVec, dLatentVec, target


class LatentDataset(Dataset):
    def __init__(self, data, train0test1, loader=DefaultLoader):
        self.data = data
        self.loader = loader
        self.train0test1 = train0test1

    def __getitem__(self, index):
        if self.train0test1 == 0:
            piece = self.data.loc[index]
        else:
            piece = self.data.loc[index+4800]
        geLatentVec, dLatentVec, target = self.loader(piece)
        return geLatentVec, dLatentVec, target

    def __len__(self):
        return len(self.data)