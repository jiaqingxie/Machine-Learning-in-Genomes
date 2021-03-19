"""
In this proram, we show that drugs similar in encoded vectors
share similar anti-cancer performance.
"""

import os
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import Normalizer

# Select a drug, and search similar drugs
DRUG_ID = 9


dataFile = os.path.join('.\\mlps_drug_exp\\data',
                        'LatentVec_Drug(unsampled).txt')
data = pd.read_csv(open(dataFile), sep=' ')

drugIds = data.loc[: , 'DRUG_ID'].tolist()

drug = data[data.DRUG_ID==DRUG_ID].values.tolist()[0][1:]
distances = {}

for i in range(0, data.shape[0]):

    temp = data.loc[i].values.tolist()[1:]
    sum = 0
    for j in range(0, 56):
        sum += (drug[j]-temp[j])**2
        pass

    sum **= 0.5
    distances[data.loc[i].values.tolist()[0]] = sum
distances[0] = 10000
distances = sorted(distances.items(), key = lambda x: x[1])


dataFile = os.path.join('.\\mlps_drug_exp\\data',
                        'LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt')
data = pd.read_csv(open(dataFile), sep='\t')

similarDrugs = [int(distances[i][0]) for i in range(0, 4)]

pieces = data.loc[data['DRUG_ID'].isin(similarDrugs)]



pass




















