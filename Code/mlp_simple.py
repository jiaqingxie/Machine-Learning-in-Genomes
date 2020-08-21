from torch import nn
import numpy as np
import torch

class MLP_simple(nn.Module):

    def __init__(self):
        super(MLP_simple, self).__init__()


        activateLayer = nn.PReLU()


        self.geMLP = nn.Sequential(
            nn.Linear(256, 256),
              
            nn.PReLU(),

            nn.Linear(256, 256),
              
            nn.PReLU(),

            nn.Linear(256, 64),
              
            nn.PReLU())  # nn.BatchNorm1d(256),


        self.dMLP = nn.Sequential(
            nn.Linear(56, 128),
              
            nn.PReLU(),

            nn.Linear(128, 128),
              
            nn.PReLU(),

            nn.Linear(128, 64),
              
            nn.PReLU())  # nn.BatchNorm1d(128),


        self.combineMLP = nn.Sequential(
            nn.Linear(128, 128),
              
            nn.PReLU(),

            nn.Linear(128, 128),
              
            nn.PReLU(),

            nn.Linear(128, 64),
              
            nn.PReLU(),
            
            nn.Linear(64, 1),
            # nn.PReLU(),
        )  # nn.BatchNorm1d(128),


    def forward(self, geLatentVec, dLatentVec):
        ge = self.geMLP(geLatentVec)
        d = self.dMLP(dLatentVec)

        combination = torch.cat([ge, d], dim=1)
        res = self.combineMLP(combination)

        return res