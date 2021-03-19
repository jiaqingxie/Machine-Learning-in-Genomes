import pandas as pd
import numpy as np


'''
chunks = pd.read_csv(path+'TCGA_breast_gene_expression.txt',sep='\t',iterator = True)
chunk = chunks.get_chunk(100000)

chunk = chunk[["Sample name"]]
chunk = chunk.drop_duplicates(keep='first')

print(chunk)
'''

# Find corresponding cancer cell lines in GDSC
'''
def CCLE2GDSC(str):
    index = str.find('_')
    res = str[0: index]
    iter = 0
    while iter<len(res) and (res[iter]<'0' or res[iter]>'9'):
        iter += 1
    strList = list(res)
    strList.insert(iter, '-')

    return ''.join(strList), str[0: index]


path = "D:\Study\CIS\数据集\CCLE\\"
df = pd.read_table(open(path + 'CCLE_gene_expression_train(breast).txt'), sep='\t')
df = df.drop('gene_id', axis=1)
df = df.drop('transcript_ids', axis=1)
CCLE_CellLines = list(df.columns.values)  # all cancer line information
Revised_CellLines = CCLE_CellLines[:]
iter = 0
for iter in range(0, len(CCLE_CellLines)):
    temp = CCLE_CellLines[iter]
    Revised_CellLines[iter], CCLE_CellLines[iter] = CCLE2GDSC(temp)

path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_table(open(path + 'GDSC1_fitted_dose_response_25Feb20.txt'), sep='\t')
df = df[['CELL_LINE_NAME', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50']]

CCLE_CellLines = set(CCLE_CellLines)
Revised_CellLines = set(Revised_CellLines)
combine = list(CCLE_CellLines | Revised_CellLines)

df = df.loc[df['CELL_LINE_NAME'].isin(combine)]

df.to_csv(path + 'GDSC1_(breast).txt', sep='\t', index=False)
'''

#
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_table(open(path + 'PUBCHEM_id_GDSC1.txt'), sep='\t')
df2 = pd.read_table(open(path + 'SMILES_GDSC1.txt'), sep='\t')

df = pd.merge(df, df2)
df.to_csv(path + 'GDSC1_DRUG_SMILES.txt', sep='\t', index=False)
'''

# breast data in GDSC incoporated with SMILES representation of drugs
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_table(open(path + 'DRUG_SMILES.txt'), sep='\t')
df2 = pd.read_table(open(path + 'GDSC1_(breast).txt'), sep='\t')
df = pd.merge(df, df2)

df = df[['CELL_LINE_NAME', 'DRUG_ID', 'SMILES_expression', 'LN_IC50']]
df.to_csv(path + 'CellLine_Smiles_IC50.txt', sep='\t', index=False)
'''

# CSI --> latent vectors of drugs
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_table(open(path + 'CellLine_Smiles_IC50.txt'), sep='\t')
df2 = pd.read_table(open(path + 'JTVAE_latent_size_60.txt'), sep='\t')
df = pd.merge(df, df2)

df = df.drop('CELL_LINE_NAME', axis=1)
df = df.drop('DRUG_ID', axis=1)
df = df.drop('SMILES_expression', axis=1)
df = df.drop('LN_IC50', axis=1)
df.to_csv(path + 'LatentVec_Drug_CSI.txt', sep='\t', index=False)
'''


# CSI --> latent vectors of drugs + latent vectors of gene expression

path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_table(open(path + 'CellLine_Smiles_IC50.txt'), sep='\t')
df2 = pd.read_table(open(path + 'JTVAE_latent_size_60.txt'), sep='\t')
df = pd.merge(df, df2, how='inner')

# df = df.drop('DRUG_ID', axis=1)
df = df.drop('SMILES_expression', axis=1)
# df = df.drop('LN_IC50', axis=1)
df["CELL_LINE_NAME"] = df.apply(lambda row: row["CELL_LINE_NAME"].replace('-', ''),axis = 1)

df2 = pd.read_table(open(path + 'LatentVec_CellLine_CSI.tsv'), sep='\t')
df = pd.merge(df, df2, on="CELL_LINE_NAME", how='inner')

# df = df.drop('CELL_LINE_NAME', axis=1)
df.to_csv(path + 'LatentVec_Drug+GeneExp_CSI.txt', sep='\t', index=False)


# Dispose of the column name
'''
path = "D:\Study\CIS\数据集\GDSC\\"
df = pd.read_table(open(path + 'LatentVec_CellLine_CSI.tsv'), sep='\t')

df.columns = df.columns.map(lambda x: x.replace('g', 'ge'))
df.to_csv(path + 'LatentVec_CellLine_CSI.tsv', sep='\t', index=False)
'''

# breast cancer cell lines
'''
df = pd.read_table(path + 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt')
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

# Hugo nomenclature
'''
Breast_Cell_lines = ['gene_id', 'transcript_ids']
df = pd.read_table(path + 'CCLE_gene_expression_train(breast).txt', sep='\t')
gene_id_raw = df[Breast_Cell_lines]
mat1 = np.array(gene_id_raw)

df2 = pd.read_table(path + 'CCLE_RNAseq_genes_rpkm_20180929.txt')
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
df = pd.read_table(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t')
#k = df.drop('gene_id', axis=1)
#k.to_csv(path + 'CCLE_gene_expression_train(v2).txt', sep='\t', index=False)

path = "D:\Study\CIS\数据集\CGC\\"
df2 = pd.read_csv(path + 'Census_allThu Jul 30 07_07_22 2020.csv')
kk = df2['Gene Symbol']
kk = np.array(kk)
kk = kk.tolist()
kk = set(kk)

#df = pd.read_table(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t')
gene = df['Description']
gene = np.array(gene)
gene = gene.tolist()
gene = set(gene)

combine = list(kk & gene)

#df2 = pd.read_table(path + 'CCLE_gene_expression_train(hugo).txt', sep='\t', index_col=0)
# print(df2['Gene'][1])

print(df.head())
ff = df.loc[df['Description'].isin(combine)]
print(ff.shape[0])
ff.to_csv(path + 'CCLE_gene_expression_train(cgc).txt', sep='\t', index=True)
'''


"""
for df in table:
    #1.对每一个分块df处理
    df_list.append(df)
    print(type(df),df.shape)
df = pd.concat(df_list, ignore_index=True)
"""


