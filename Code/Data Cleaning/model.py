import torch 
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim

import pandas as pd
import numpy as np

def get_df(file):
    mylist = []
    for chunk in  pd.read_csv(file,sep='\t',chunksize=20000):
        mylist.append(chunk)
    temp_df = pd.concat(mylist, axis= 0)
    del mylist
    return temp_df
    




class OurModel(nn.Module):
    def __init__(self,feature_number):
        super(OurModel,self).__init__()
        #pretrained encoder
        self.feature_number = feature_number
        self.fc1 = nn.Linear(self.feature_number,256)
        self.fc2 = nn.Linear(256,64)
        self.fc31 = nn.Linear(64,16)   
        self.fc31 = nn.Linear(64,16)
    
    def encode(self,x):
        pass
    
    def reparameterize(self, mu, logvar):
        pass

    def decode(self,x):
        pass

    def forward(self,x):
        pass







if __name__ == "__main__":
    #fd = pd.read_table('CCLE_RNAseq_rsem_transcripts_tpm_20180929.txt')
    #fd = pd.read_csv('TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09 (1).tsv', sep = '\t',chunksize = 20000)
    #fd = get_df('TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09 (1).tsv')

    #fd =  pd.read_excel('GDSC1_fitted_dose_response_25Feb20.xlsx')
    #print(fd['CELL_LINE_NAME'].value_counts())
    fd = get_df('CosmicCLP_MutantExport.tsv')
    print(fd['GENE_MUTATION_ID'].value_counts()) #counting the numbers of drugs(no repeat)
    #print(fd.shape[0]) #numbers of rows
    #print(fd.shape[1]) #numbers of colomns

    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    

   # model = VAE().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)




    
    



    
