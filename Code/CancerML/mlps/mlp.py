from torch import nn
import numpy as np
import torch

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.geLayer1 = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.geLayer2 = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.geLayer3 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(True))

        self.dLayer1 = nn.Sequential(nn.Linear(60, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.dLayer2 = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.dLayer3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(True))

        self.combineLayer1 = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.combineLayer2 = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.combineLayer3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(True))
        self.combineLayer4 = nn.Sequential(nn.Linear(64, 1))

    def forward(self, geLatentVec, dLatentVec):
        # ge = np.array(geLatentVec)
        # ge = torch.from_numpy(ge)
        ge = self.geLayer1(geLatentVec)
        ge = self.geLayer2(ge)
        ge = self.geLayer3(ge)

        # d = np.array(dLatentVec)
        # d = torch.from_numpy(d)
        d = self.dLayer1(dLatentVec)
        d = self.dLayer2(d)
        d = self.dLayer3(d)

        combination = torch.cat([ge, d], dim=1)
        combination = self.combineLayer1(combination)
        combination = self.combineLayer2(combination)
        combination = self.combineLayer3(combination)
        res = self.combineLayer4(combination)

        return res