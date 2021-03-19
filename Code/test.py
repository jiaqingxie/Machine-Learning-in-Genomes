
import pandas as pd
from sklearn.preprocessing import Normalizer
data = pd.DataFrame({'a':[1,2,3],'b':[4,5,6],'c':[7,8,9]})
data = data.T
data = data.T
print(data)
pass



'''
import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import r2_score
import pandas as pd

from mlps_drug_exp.mlp_batchnorm import MLP_batchnorm
from mlps_drug_exp.lantentDataset import LatentDataset, DefaultLoader

model = MLP_batchnorm()
if torch.cuda.is_available():
    model = model.cuda()

path = '.\\trainedModels\\'
model.load_state_dict(torch.load(path + 'modelParameters.pt'))
print(model.state_dict()['geMLP.1.running_mean'][:5])
print('-----------------------------------------------')

model.train()



dataFile = os.path.join('.\\data', 'LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt')
data = pd.read_csv(open(dataFile), sep='\t')

testData = data.loc[3165: 3329, :]
testDataset = LatentDataset(testData, train0val1test2=2)
testLoader = DataLoader(testDataset, batch_size=165, drop_last=True)

criterion = nn.MSELoss()
for batch in testLoader:
    geLatentVec, dLatentVec, target = batch
    if torch.cuda.is_available():
        geLatentVec = geLatentVec.cuda()
        dLatentVec = dLatentVec.cuda()
        target = target.cuda()

    out = model(geLatentVec, dLatentVec)
    loss = criterion(out, target)
    out = out.data.cpu().numpy().tolist()
    target = target.cpu().numpy().tolist()
    print(out)
    print(target)
    print(model.state_dict()['geMLP.1.running_mean'][:5])
    print('-----------------------------------------------')

    model.eval()
    geLatentVec1, dLatentVec1, target1 = batch
    if torch.cuda.is_available():
        geLatentVec1 = geLatentVec1.cuda()
        dLatentVec1 = dLatentVec1.cuda()
        target1 = target1.cuda()

    out1 = model(geLatentVec1, dLatentVec1)
    loss1 = criterion(out1, target1)
    out1 = out1.data.cpu().numpy().tolist()
    target1 = target1.cpu().numpy().tolist()
    print(out1)
    print(target)
    print(model.state_dict()['geMLP.1.running_mean'][:5])
    diff = [out[i][j]-out1[i][j] for j in range(len(out[0])) for i in range(len(out))]
    print(sum(diff))
    pass



    model.train()
    geLatentVec, dLatentVec, target = batch
    if torch.cuda.is_available():
        geLatentVec = geLatentVec.cuda()
        dLatentVec = dLatentVec.cuda()
        target = target.cuda()

    out = model(geLatentVec, dLatentVec)
    loss = criterion(out, target)
    out = out.data.cpu().numpy().tolist()
    target = target.cpu().numpy().tolist()
    print(out)
    print(target)
    print(model.state_dict()['geMLP.1.running_mean'][:5])
    print('-----------------------------------------------')

    model.eval()
    geLatentVec1, dLatentVec1, target1 = batch
    if torch.cuda.is_available():
        geLatentVec1 = geLatentVec1.cuda()
        dLatentVec1 = dLatentVec1.cuda()
        target1 = target1.cuda()

    out1 = model(geLatentVec1, dLatentVec1)
    loss1 = criterion(out1, target1)
    out1 = out1.data.cpu().numpy().tolist()
    target1 = target1.cpu().numpy().tolist()
    print(out1)
    print(target)
    print(model.state_dict()['geMLP.1.running_mean'][:5])
    diff = [out[i][j] - out1[i][j] for j in range(len(out[0])) for i in range(len(out))]
    print(sum(diff))
    pass
'''


