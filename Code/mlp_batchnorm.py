from torch import nn
import numpy as np
import torch

class MLP_batchnorm(nn.Module):

    def __init__(self):
        super(MLP_batchnorm, self).__init__()
        '''
        self.geLayer1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True))# nn.BatchNorm1d(256),
        self.geLayer2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True))# nn.BatchNorm1d(256),
        self.geLayer3 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(True))

        self.dLayer1 = nn.Sequential(nn.Linear(56, 128), nn.ReLU(True))# nn.BatchNorm1d(128),
        self.dLayer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True))# nn.BatchNorm1d(128),
        self.dLayer3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(True))

        self.combineLayer1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True))# nn.BatchNorm1d(128),
        self.combineLayer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True))# nn.BatchNorm1d(128),
        self.combineLayer3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(True))# nn.BatchNorm1d(64),
        self.combineLayer4 = nn.Sequential(nn.Linear(64, 1))
        '''

        activateLayer = nn.PReLU()


        self.geMLP = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2),
            nn.PReLU())  # nn.BatchNorm1d(256),


        self.dMLP = nn.Sequential(
            nn.Linear(56, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.PReLU())  # nn.BatchNorm1d(128),


        self.combineMLP = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.PReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.PReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.PReLU(),

            nn.Linear(64, 1),
            # nn.PReLU(),
        )  # nn.BatchNorm1d(128),


    def forward(self, geLatentVec, dLatentVec):
        # ge = np.array(geLatentVec)
        # ge = torch.from_numpy(ge)
        # geLatentVec = geLatentVec.unsqueeze(1)
        ge = self.geMLP(geLatentVec)

        # d = np.array(dLatentVec)
        # d = torch.from_numpy(d)
        # dLatentVec = dLatentVec.unsqueeze(1)
        d = self.dMLP(dLatentVec)


        # combination = torch.cat([ge, d], dim=2)
        # combination = combination.squeeze(1)

        # d = d.squeeze(1)
        combination = torch.cat([ge, d], dim=1)
        res = self.combineMLP(combination)

        return res