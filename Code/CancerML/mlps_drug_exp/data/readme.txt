LatentVec_Drug(unsampled): 
Latent vectors of drug molecular data encoded by JTVAE model. 
"Unsampled" means we use the mean value of the distribution given by JTVAE. 

LatentVec_Drug+GeneExp(breast+eliminated+unsampledGene+unsampledDrug): 
Latent vectors of drug molecular data encoded by JTVAE model and breast cancer cell line gene expression data encoded by geneVAE model. 
"Eliminated" means we eliminate gene expression entrances which show low relevance with cancer. 

LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug): 
The content is similar with the former one, but cancer cell line gene expression data is filtered by CGC dataset before being encoded by geneVAE. 

LatentVec_Drug+RawVec_GeneExp(cgc+eliminated+unsampledDrug): 
In this file, cancer cell line gene expression data is filtered by CGC dataset but not encoded by geneVAE. 

Pancancer_LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug): 
This file is generated on pan cancer dataset. 