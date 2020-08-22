import os
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import Normalizer


dataFile = os.path.join('E:\Machine Learning in Genomics\mlps_drug_exp\data',
                        'LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt')
data = pd.read_csv(open(dataFile), sep=' ')

drugIds = data.loc[: , 'DRUG_ID'].tolist()

drug = data[data.DRUG_ID==60].values.tolist()[0][1:]
distances = {}

for i in range(0, data.shape[0]):
    if i not in drugIds:
        continue
    temp = data.loc[data['DRUG_ID']==i].values.tolist()[0][1:]
    sum = 0
    for j in range(0, 56):
        sum += (drug[0]-temp[0])**2
        pass

    sum **= 0.5
    distances[str(i)] = sum

distances = sorted(distances.items(), key = lambda x: x[1])























