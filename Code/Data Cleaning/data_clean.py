import BruteForce
import pandas as pd 
import numpy as np
if __name__ == "__main__":
    
    df = pd.read_table('CCLE_gene_expression.txt')
    Cell_lines = list(df.columns.values) #all cancer line information
    Breast_Cell_lines = ['gene_id','transcript_ids']
    
    #Other Cell Lines....
    for i in range(len(Cell_lines)):
        if BruteForce.BF(Cell_lines[i],'BREAST'):   # BREAST pattern match
            Breast_Cell_lines.append(Cell_lines[i])
    
    print(Breast_Cell_lines) #see all the breast cancer lines
    print(len(Breast_Cell_lines))

    f = df[Breast_Cell_lines]
    f.to_csv('CCLE_gene_expression_train(v1).txt', sep='\t',index=False)
    
    #-----------------------------   ---------------------------------#

    Breast_Cell_lines = ['gene_id','transcript_ids']
    df = pd.read_table('CCLE_gene_expression_train(v1).txt',sep = '\t')
    gene_id_raw = df[Breast_Cell_lines]
    mat1 = np.array(gene_id_raw)
    
    df2 = pd.read_table('CCLE_RNAseq_genes_rpkm_20180929.txt')
    Breast_Cell_lines = ['Name','Description']
    gene_id_done = df2[Breast_Cell_lines]
    mat2 = np.array(gene_id_done)


    for j in range(mat1.shape[0]):
        print(j)
        for i in range(mat2.shape[0]):
            if mat1[j][0] == mat2[i][0]:
                mat1[j][1] = mat2[i][1]
                #print("1")
                break
            else:
                mat1[j][1] = 0
        
    mat1 = pd.DataFrame(mat1)
    df['transcript_ids'] = mat1[1]
    print(df.head())
    df.to_csv('CCLE_gene_expression_train(v2).txt', sep='\t',index=False)
    

    df = pd.read_table('CCLE_gene_expression_train(v2).txt', sep = '\t')
    k = df.drop('gene_id',axis = 1)
    k.to_csv('CCLE_gene_expression_train(v2).txt', sep = '\t',index=False)
    
    df = pd.read_csv('Census_allThu Jul 30 07_07_22 2020.csv')
    kk = df['Gene Symbol']
    kk = np.array(kk)
    kk = kk.tolist()
    kk = set(kk)

    df = pd.read_table('CCLE_gene_expression_train(v2).txt', sep = '\t')
    gene = df['Gene']
    gene = np.array(gene)
    gene = gene.tolist()
    gene = set(gene)

    combine = list(kk & gene)
    
    df2 = pd.read_table('CCLE_gene_expression_train(v2).txt', sep = '\t', index_col = 0)
    #print(df2['Gene'][1])
    
    print(df2.head())
    ff = df2.loc[combine]
    print(ff.shape[0])
    ff.to_csv('CCLE_gene_expression_train(v3).txt', sep='\t',index=True)