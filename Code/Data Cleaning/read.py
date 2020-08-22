import pandas as pd 
import numpy as np
import seaborn as sns
'''
df = pd.read_table('CCLE_gene_expression_train(v1).txt')
ks = df.drop('gene_id',axis = 1)
ks = ks.drop('transcript_id',axis = 1)

data = np.array(ks.loc[:,:])
ax = sns.heatmap(data[0:100,:])

dd = pd.read_table('gene_attribute_matrix.txt')
print(dd.head())
'''
df = pd.read_table('TCGA_breast_gene_expression.txt')
print(df.head())
cell_line = ['TCGA-AR-A5QQ-01','TCGA-D8-A1JA-01','TCGA-BH-A0BQ-01','TCGA-BH-A0BT-01','TCGA-A8-A06X-01','TCGA-A8-A096-01','TCGA-BH-A0C7-01','TCGA-AC-A5XU-01']
ks = df[cell_line]
#ks = ks.drop('sample', axis = 1)
data = np.array(ks.loc[:,:])
ax = sns.heatmap(data[0:100,:])