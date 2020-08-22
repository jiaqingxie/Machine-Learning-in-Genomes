import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
dataset = pd.read_csv('LatentVec_Drug+GeneExp_CSI(1).txt',sep = '\t')

input = dataset.iloc[:,3:319]
train_input = dataset.iloc[0:5800,3:319]
test_input = dataset.iloc[5800:5820,3:319]

print(train_input.head(5))

output = dataset.iloc[:,2]
train_output = dataset.iloc[0:5800,2]
test_output = dataset.iloc[5800:5820,2]




input = np.array(input)
output = np.array([output])
train_input = np.array(train_input)
test_input = np.array(test_input)
train_output = np.array(train_output)
test_output = np.array(test_output)



output = output.T
train_output = train_output.T 
test_output = test_output.T


input = torch.FloatTensor(input).cuda()
output = torch.FloatTensor(output).cuda()
train_input = torch.FloatTensor(train_input).cuda()
test_input = torch.FloatTensor(test_input).cuda()
train_output = torch.FloatTensor(train_output).cuda()
test_output = torch.FloatTensor(test_output).cuda()



class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)   
        self.linear_2 = nn.Linear(n_hidden, 32)
        self.out = nn.Linear(32, n_output)   

    def forward(self, x):
        x = F.elu(self.hidden(x))      
        x = F.dropout(x)
        x = F.elu(self.linear_2(x))
        x = F.dropout(x)
        x = self.out(x)
        return x

net = Net(n_feature=312, n_hidden=200, n_output=1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.013)
loss_func = torch.nn.MSELoss()
#loss_func = torch.nn.CrossEntropyLoss()

net.cuda()
print(input[0:2])


for i in range(2000):
    out = net(input)       
    # out = net(input[0:5600])            
    #loss = loss_func(out, train_output)   
    loss = loss_func(out, output) 
    print(loss) 
    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()        

with torch.no_grad():
    out = net(input)
    #out = net(test_input)
    print(out[0:10])
    print(output[0:10])
    out = out.cpu().numpy()
    output = output.cpu().numpy()
    #test_output = test_output.cpu().numpy()
    print(r2_score(output,out))
    plt.scatter(out,output)