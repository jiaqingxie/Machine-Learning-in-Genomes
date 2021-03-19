import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

from mlps_drug_exp.mlp_batchnorm import MLP_batchnorm
from mlps_drug_exp.lantentDataset import LatentDataset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

model = MLP_batchnorm()
if torch.cuda.is_available():
    model = model.cuda()

path = '.\\trainedModels\\'
model.load_state_dict(torch.load(path + 'modelParameters.pt'))


dataFile = os.path.join('.\\data', 'LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt')
data = pd.read_table(open(dataFile))
data = data.sample(frac=1).reset_index(drop=True)

testData = data.loc[3000: , :]
testDataset = LatentDataset(testData, train0val1test2=1)
testLoader = DataLoader(testDataset, batch_size=300, drop_last=True)


model.eval()
criterion = nn.MSELoss()
# eval_acc = 0
for batch in testLoader:
    geLatentVec, dLatentVec, target = batch
    if torch.cuda.is_available():
        geLatentVec = geLatentVec.cuda()
        dLatentVec = dLatentVec.cuda()
        target = target.cuda()

    out = model(geLatentVec, dLatentVec)
    loss = criterion(out, target)
    evalLoss = loss.data.item()
    # SS_tot = torch.std(target)
    # SS_res = evalLoss
    out = out.data.cpu().numpy().tolist()
    target = target.cpu().numpy().tolist()
    r2 = r2_score(target, out)
    print('Test Loss: {:.6f}, R2_Score: {:.6f}'.format(evalLoss, r2))

    with torch.no_grad():
        plt.scatter(target, out)

# print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

