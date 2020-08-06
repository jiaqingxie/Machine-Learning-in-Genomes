import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from sklearn.metrics import r2_score
import pandas as pd

from mlps.mlp import MLP
from mlps.lantentDataset import LatentDataset

batch_size = 64
learning_rate = 0.01
num_epoches = 500

data = pd.read_table(open('D:\Study\CIS\数据集\GDSC\LatentVec_Drug+GeneExp_CSI.txt'))
trainData = data.loc[: 4800, :]
trainDataset = LatentDataset(trainData, train0test1=0)
testData = data.loc[4800: , :]
testDataset = LatentDataset(testData, train0test1=1)

trainLoader = DataLoader(trainDataset, batch_size=50, drop_last=True)
testLoader = DataLoader(testDataset, batch_size=1000, drop_last=True)


model = MLP()

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


epoch = 0
bestLoss = 200
path = 'D:\Study\CIS\CancerML\\trainedModels\\'
while epoch < 2000:
    for batch in trainLoader:
        geLatentVec, dLatentVec, target = batch

        # if geLatentVec.shape[0] != 50:
        #     continue

        if torch.cuda.is_available():
            geLatentVec = geLatentVec.cuda()
            dLatentVec = dLatentVec.cuda()
            target = target.cuda()
        else:
            geLatentVec = Variable(geLatentVec)
            dLatentVec = Variable(dLatentVec)
            target = Variable(target)
        out = model(geLatentVec, dLatentVec)
        loss = criterion(out, target)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch += 1
    if epoch % 2 == 0:

        model.eval()
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

            print('epoch: {}, Test Loss: {:.6f}, R2_Score: {:.6f}'.format(epoch, evalLoss, 1 - SS_res/SS_tot))
            if (evalLoss < bestLoss):
                bestLoss = evalLoss
                torch.save(model.state_dict(), path + 'modelParameters.pt')
                print("Got a better model!")
        # print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))



    pass









