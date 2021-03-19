"""
This file preprocesses the CCLE gene expression dataset to match the cancer cell lines
in GDSC dataset, and combine the encoded vectors of drug molecular data,cancer cell line
gene expression data and the corresponding lnIC50 value in a single file to train the MLP model.
"""

import pandas as pd
import numpy as np


#-------------------------------------------EXPRESSION------------------------------------------------------------------
# breast cancer cell lines
'''
df = pd.read_csv(path + 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt', sep='\t')
Cell_lines = list(df.columns.values)  # all cancer line information
Breast_Cell_lines = ['gene_id', 'transcript_ids']

# Other Cell Lines....
for i in range(len(Cell_lines)):
    if Cell_lines[i].find( 'BREAST') != -1:  # BREAST pattern match
        Breast_Cell_lines.append(Cell_lines[i])

print(Breast_Cell_lines)  # see all the breast cancer lines
print(len(Breast_Cell_lines))

f = df[Breast_Cell_lines]
f.to_csv(path + 'CCLE_gene_expression_train(breast).txt', sep='\t', index=False)
'''




# Find corresponding cancer cell lines in GDSC
'''
def CCLE2Standard(str):
    index = str.find('_')
    res = str[0: index]
    return res


path = "D:\Study\CIS\数据集\CCLE\\"
df = pd.read_csv(open(path + 'CCLE_gene_expression_train(breast).txt'), sep='\t')
df = df.drop('gene_id', axis=1)
df = df.drop('transcript_ids', axis=1)
CCLE_CellLines = list(df.columns.values)  # all cancer line information
Standard_CellLines = CCLE_CellLines[:]
iter = 0
for iter in range(0, len(CCLE_CellLines)):
    temp = CCLE_CellLines[iter]
    Standard_CellLines[iter] = CCLE2Standard(temp)

path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_csv(open(path + 'GDSC1_fitted_dose_response_25Feb20.txt'), sep='\t')
df = df[['CELL_LINE_NAME', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50']]

Standard_CellLines = set(Standard_CellLines)


df = df.loc[df['CELL_LINE_NAME'].replace('-', '').isin(Standard_CellLines)]

df.to_csv(path + 'GDSC1_(breast).txt', sep='\t', index=False)
'''

# Hugo nomenclature
'''
Breast_Cell_lines = ['gene_id', 'transcript_ids']
df = pd.read_csv(path + 'CCLE_gene_expression_train(breast).txt', sep='\t')
gene_id_raw = df[Breast_Cell_lines]
mat1 = np.array(gene_id_raw)

df2 = pd.read_csv(path + 'CCLE_RNAseq_genes_rpkm_20180929.txt')
Breast_Cell_lines = ['Name', 'Description']
gene_id_done = df2[Breast_Cell_lines]
mat2 = np.array(gene_id_done)

for j in range(mat1.shape[0]):
    # print(j)
    for i in range(mat2.shape[0]):
        if mat1[j][0] == mat2[i][0]:
            mat1[j][1] = mat2[i][1]
            # print("1")
            break
        else:
            mat1[j][1] = 0

mat1 = pd.DataFrame(mat1)
df['transcript_ids'] = mat1[1]
df.rename(columns={'transcript_ids':'Description'},inplace=True)
print(df.head())
df.to_csv(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t', index=False)
'''

# Intersection with CGC
'''
df = pd.read_csv(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t')

path = "D:\Study\CIS\数据集\CGC\\"
df2 = pd.read_csv(path + 'Census_allThu Jul 30 07_07_22 2020.csv')
kk = df2['Gene Symbol']
kk = np.array(kk)
kk = kk.tolist()
kk = set(kk)

#df = pd.read_csv(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t')
gene = df['Description']
gene = np.array(gene)
gene = gene.tolist()
gene = set(gene)

combine = list(kk & gene)

#df2 = pd.read_csv(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t', index_col=0)
# print(df2['Gene'][1])

print(df.head())
ff = df.loc[df['Description'].isin(combine)]
print(ff.shape[0])
ff.to_csv(path + 'CCLE_gene_expression_train(cgc).txt', sep='\t')
'''


# eliminate useless genes
'''
from numpy import *
path = "D:\Study\CIS\数据集\CCLE\\"
df = pd.read_csv(path + 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt', sep='\t')
df.drop('transcript_ids', axis=1, inplace=True)
print(df.shape)
dfList = []

for i in df.index:
    rowData = df.loc[i].values[:]
    rowData = rowData.tolist()
    if std(rowData[1:])<0.5:
        continue
    dfList.append(rowData)

df = pd.DataFrame({dfList[i][0]: dfList[i][1: ] for i in range(0, len(dfList))})
df = df.T

df2 = pd.read_csv(path + 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt', sep='\t')
df.columns = df2.columns.values[2: ]

df.to_csv(path + 'CCLE_gene_expression_train(eliminated).txt', sep='\t', index=False)
'''




# Additional process
'''
path = "D:\Study\CIS\数据集\CCLE\\"
df = pd.read_csv(path + 'CCLE_gene_expression_train(cgc).txt', sep='\t')
df = df.drop('num', axis=1)
# df = df.drop('gene_id', axis=1)
df.to_csv(path + 'CCLE_gene_expression_train(cgc).txt', sep='\t', index=False)
'''


# steven eliminate v4
'''
path = "C:\\Users\dhy\Documents\WeChat Files\wxid_2isdora2arxe12\FileStorage\File\\2020-08\\"
df = pd.read_csv(path + 'CCLE_gene_expression_train_pancancer(v4)(1).txt', sep='\t')
df2 = pd.read_csv(path + 'Pancancer_Name(1).txt', sep=' ')

cellLines = df.columns.values

for i in range(1, df.shape[1]):
    if df2.loc[df.loc[641, cellLines[i]], 'num'] < 30:
        df.drop(cellLines[i], axis=1, inplace=True)


df.to_csv(path + 'CCLE_gene_expression_train(v4+eliminated_for_tsne_30).txt', sep='\t', index=False)
'''














#-------------------------------------------DRUG_RESPONSE---------------------------------------------------------------
#
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_csv(open(path + 'PUBCHEM_id_GDSC1.txt'), sep='\t')
df2 = pd.read_csv(open(path + 'SMILES_GDSC1.txt'), sep='\t')

df = pd.merge(df, df2)
df.to_csv(path + 'GDSC1_DRUG_SMILES.txt', sep='\t', index=False)
'''

# breast data in GDSC incoporated with SMILES representation of drugs
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_csv(open(path + 'DRUG_SMILES.txt'), sep='\t')
df2 = pd.read_csv(open(path + 'GDSC1_(pan).txt'), sep='\t')
df = pd.merge(df, df2)

df = df[['CELL_LINE_NAME', 'DRUG_ID', 'SMILES_expression', 'LN_IC50']]
df.to_csv(path + 'CellLine_Smiles_IC50(pan).txt', sep='\t', index=False)
'''


# CSI --> latent vectors of drugs + gene expression
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_csv(open(path + 'CellLine_Smiles_IC50.txt'), sep='\t')
df2 = pd.read_csv(open(path + 'LatentVec_Drug(unsampled).txt'), sep=' ')
df = pd.merge(df, df2, how='inner')

# df = df.drop('DRUG_ID', axis=1)
df = df.drop('SMILES_expression', axis=1)
# df = df.drop('LN_IC50', axis=1)
df["CELL_LINE_NAME"] = df.apply(lambda row: row["CELL_LINE_NAME"].replace('-', ''),axis = 1)

df2 = pd.read_csv(open(path + 'RawVec_GeneExp(cgc+eliminated).tsv'), sep='\t')
df = pd.merge(df, df2, on="CELL_LINE_NAME", how='inner')


# df = df.drop('CELL_LINE_NAME', axis=1)
df.to_csv(path + 'LatentVec_Drug+RawVec_GeneExp(cgc+eliminated+unsampledDrug).txt', sep='\t', index=False)
'''


# CSI --> latent vectors of drugs + gene expression + gene mutation
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_csv(open(path + 'CellLine_Smiles_IC50.txt'), sep='\t')
df2 = pd.read_csv(open(path + 'LatentVec_Drug.txt'), sep=' ')
df = pd.merge(df, df2, how='inner')

# df = df.drop('DRUG_ID', axis=1)
df = df.drop('SMILES_expression', axis=1)
# df = df.drop('LN_IC50', axis=1)
df["CELL_LINE_NAME"] = df.apply(lambda row: row["CELL_LINE_NAME"].replace('-', ''),axis = 1)

df2 = pd.read_csv(open(path + 'LatentVec_GeneExp.tsv'), sep='\t')
df = pd.merge(df, df2, on="CELL_LINE_NAME", how='inner')
df2 = pd.read_csv(open(path + 'LatentVec_GeneMut.tsv'), sep='\t')
df = pd.merge(df, df2, on="CELL_LINE_NAME", how='inner')

# df = df.drop('CELL_LINE_NAME', axis=1)
df.to_csv(path + 'LatentVec_Drug+GeneExp+GeneMut.txt', sep='\t', index=False)
'''


# Dispose of the column name
'''
path = "D:\Study\CIS\数据集\CCLE\\"
df = pd.read_csv(open(path + 'RawVec_GeneExp(cgc+eliminated).tsv'), sep='\t')

df.columns = df.columns.map(lambda x: 'ge'+x)
df.to_csv(path + 'RawVec_GeneExp(cgc+eliminated).tsv', sep='\t', index=False)
'''


# Dispose of the row name
'''
path = "D:\Study\CIS\数据集\CCLE\\"
df2 = pd.read_csv(open(path + 'CCLE_gene_expression_train(cgc+eliminated).txt'), sep='\t')
path = "C:\\Users\dhy\Desktop\新建文件夹\\"
df = pd.read_csv(open(path + 'LatentVec_GeneExp_normbyfeature(cgc+eliminated+unsampled).tsv'), sep='\t')

for i in range(0, df.shape[0]):
    df.loc[i, 'CELL_LINE_NAME'] = df2.columns.values[i].split('_')[0]

df.to_csv(path + 'LatentVec_GeneExp_normbyfeature(cgc+eliminated+unsampled).tsv', sep='\t', index=False)
'''

# Raw Vecs
'''
path = "D:\Study\CIS\数据集\CCLE\\"
df = pd.read_csv(open(path + 'RawVec_GeneExp(cgc+eliminated).tsv'), sep='\t')
df2 = pd.read_csv(open(path + 'CCLE_gene_expression_train(cgc+eliminated).txt'), sep='\t')
cellLines = df2.columns.values
cellLines = cellLines.tolist()

# df.drop('CELL_LINE_NAME', axis=1, inplace=True)
for i in range(len(cellLines)):
    cellLines[i] = cellLines[i].split('_')[0]

df.insert(0, "CELL_LINE_NAME", cellLines)

df.to_csv(path + 'RawVec_GeneExp(cgc+eliminated).tsv', sep='\t', index=False)
'''


# Final Result
'''
from numpy import *
df = pd.read_csv(open('D:\Study\CIS\CancerML\mlps_drug_exp\\R2_Score(cgc+eliminated+unsampledGene+unsampledDrug).txt'), sep='\t')
result = mean(df.loc[16: , 'R2_test'])
print(result)
pass
'''


# Restore data file
'''
import os
dataFile = os.path.join('D:\Study\CIS\CancerML\mlps_drug_exp\\data',
                          'LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt')
df = pd.read_csv(open(dataFile), sep='\t')
for i in range(57):
    df.drop(str(i), axis=1,  inplace=True)

df.drop('Unnamed: 0', axis=1,  inplace=True)
df.to_csv(dataFile, sep='\t')
'''
