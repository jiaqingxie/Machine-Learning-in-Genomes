"""
This code is used to generate drug latent vectors which are effective
for given cancer cell line gene expression profile.
"""

import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import r2_score
import pandas as pd


from mlps_drug_exp.mlp import MLP_simple, MLP_batchnorm
from mlps_drug_exp.lantentDataset import LatentDataset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


model = MLP_simple()
if torch.cuda.is_available():
    model = model.cuda()
path = '.\\trainedModels\\'
model.load_state_dict(torch.load(path + 'modelParameters.pt'))


# Data file of encoded vectors of drug and gene expression data
dataFile = os.path.join('.\\data', 'LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt')
data = pd.read_csv(open(dataFile), sep='\t')

testData = data.loc[: 2999, :]
testDataset = LatentDataset(testData, train0val1test2=0)
testLoader = DataLoader(testDataset, batch_size=300, drop_last=True)


model.eval()
criterion = nn.MSELoss()
# eval_acc = 0
for batch in testLoader:


    num=0
    while num < 10:
        geLatentVec, dLatentVec, target = batch

        # Use HCC1187 as target cancer cell line
        for i in range(len(geLatentVec)):
            geLatentVec[i] = geLatentVec[0]

        # Generate drug latent vectors where varialbles are i.i.d. and conform N(0, 7)
        means = torch.IntTensor([0 for i in range(56)])
        stds = torch.IntTensor([7 for i in range(56)])
        for i in range(len(dLatentVec)):
            torch.normal(mean=means, std=stds, out=dLatentVec[i])


        if torch.cuda.is_available():
            geLatentVec = geLatentVec.cuda()
            dLatentVec = dLatentVec.cuda()
            target = target.cuda()

        out = model(geLatentVec, dLatentVec)
        out = out.data.cpu().numpy().tolist()

        dLatentVec = dLatentVec.data.cpu().numpy().tolist()

        for i in range(len(out)):
            if out[i][0] < -1:
                num += 1
                piece = {'LN_IC50': out[i][0]}

                for j in range(56):
                    piece['d'+str(j)] = dLatentVec[i][j]


                resultFile = os.path.join('.\\results',
                                        'LNIC50_dLatentVec_HCC1187.txt')
                results = pd.read_csv(open(resultFile), sep='\t')
                piece['id'] = int(len(results))
                results = results.append(piece, ignore_index=True)
                results.to_csv(resultFile, sep='\t', index=False)

                if num > 9:
                    break

    break
# print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))









