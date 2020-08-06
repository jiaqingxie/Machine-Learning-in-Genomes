import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

from mlps.mlp import MLP
from mlps.lantentDataset import LatentDataset


model = MLP()
if torch.cuda.is_available():
    model = model.cuda()

path = 'D:\Study\CIS\CancerML\\trainedModels\\'
model.load_state_dict(torch.load(path + 'modelParameters.pt'))


data = pd.read_table(open('D:\Study\CIS\数据集\GDSC\LatentVec_Drug+GeneExp_CSI.txt'))
testData = data.loc[4800: , :]
testDataset = LatentDataset(testData, train0test1=1)
testLoader = DataLoader(testDataset, batch_size=1000, drop_last=True)


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
    SS_tot = torch.std(target)
    SS_res = evalLoss
    print('Test Loss: {:.6f}, R2_Score: {:.6f}'.format(evalLoss, 1 - SS_res/SS_tot))

# print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

