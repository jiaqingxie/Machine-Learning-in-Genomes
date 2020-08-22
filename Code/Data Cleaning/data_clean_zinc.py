import numpy as np 
import pandas as pd 

read_1 = pd.read_csv('GDSC1_fitted_dose_response_25Feb20.csv')
read_2 = pd.read_csv('Drug_listThu Jul 30 04_25_49 2020.csv')
print(read_1.head())
print(read_2.head(5))
Drug_id = read_1['DRUG_ID']
drug_id = read_2['drug_id']
#print(set(Drug_id))

PUBCHEM_ID = []
DRUG_ID = []

k = 0
for i in range(len(list(read_2['Sample_Size']))):
    
    if read_2.iloc[i,6] == "GDSC1":
        if read_2.iloc[i,5]== read_2.iloc[i,5] and read_2.iloc[i,5]!='none' and read_2.iloc[i,5]!='several':
            k = k + 1
            PUBCHEM_ID.append(read_2.iloc[i,5])
            DRUG_ID.append(read_2.iloc[i,0])
print(k)
print(len(PUBCHEM_ID))
#print(PUBCHEM_ID)
            
data = pd.DataFrame( PUBCHEM_ID,DRUG_ID)
print(data)

data.to_csv('PUBCHEM_id_GDSC1.txt',sep='\t', header =['PUBCHEM_ID'], index = True)
            
read_3 = pd.read_table('3458529904100544365.txt',sep = '\t')
read_3.to_csv('SMILES_GDSC1.txt',sep = '\t',header = ['PUBCHEM_ID','SMILES_expression'],index = None)




