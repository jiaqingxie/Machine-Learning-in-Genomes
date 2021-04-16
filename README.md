
![](https://img.shields.io/badge/language-python-orange.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v1.0.0-519dd9.svg)
![99%](https://progress-bar.dev/99)
# Variational Autoencoders for Anti-cancer Drug Response Prediction
Welcome to my first cancer machine learning project. It's my second years' undergraduate research. Here's my teammates: Yuan and Varus and Dexion. We are supervised by Prof. Manolis Kellis from MIT CSAIL LAB. Our paper link:(https://arxiv.org/abs/2008.09763), accepted by 2021 ICLR AI4PH workshop (paper 33)



## Check your environment
You should have a pytorch version of 1.5.0 or above. Also you should have a cuda version of 9 or above.

The following order in the command line can download the newest version of pytorch and torchvision with cuda version of 10.2. 
```Bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
In /Code/CancerML/mlps_drug_exp env, run train.py to test our model on breast cancer dataset(dont need to change anything)
```python
python train.py
```
run train_pan.py to test our model on pancancer datasets
```python
python train_pan.py
```

## Introductions
With the development of molecular biology, the study of cancer genomics has enabled scientists to develop anti-cancer drugs according to cancers’ genomics features. These drugs are used widely and have great significance in the therapy of cancer treatment nowadays. However,the efficacy of anti-cancer drugs vary greatly from one kind of tumor to another,making it considerably difficult to customize therapy strategy for patients. Moreover, the efficacy of anti-cancer drugs is closely related with their molecular structure, which is also hard to predict even for sophisticated pharmacists. To provide more precise treatment strategy, a sufficient analysis and understanding of cancers’ genomic data and drugs molecular structure is essential.

Specifically, in this research, we select breast cancer, which is the main cancer in women group for our study. Many drug efficacy predictions were under the background of breast cancer but only a few studies are efficiently making the full use of encoded information to make predictions. We concentrate on judging encoders’ efficiency on extracting features. To extract features from a large amount of gene data, we propose VAE (Variational Autoencoder), which has achieved great success in the field of unsupervised learning of complex probability distribution. The amazing ability of VAE models to capture probabilistic distribution of latent information could enable more complete analysis of gene data, making it easier to predict the response of anti-cancer drugs when they are used in this specific cancer cell line. As for anti-cancer drugs, we will transform their molecular graph into junction trees, and implement a junction tree VAE model to extract their low dimensional features. Finally, we will implement a fully connected neural network to combine the extracted features to produce the final result, $IC_{50}$ value of the anti-cancer drug used against the cancer cell line.

## Related works
* [Cancer Drug Response Profile scan (CDRscan): A Deep Learning Model That Predicts Drug Effectiveness from Cancer Genomic Signature](https://www.researchgate.net/publication/325696059_Cancer_Drug_Response_Profile_scan_CDRscan_A_Deep_Learning_Model_That_Predicts_Drug_Effectiveness_from_Cancer_Genomic_Signature)


### Gene Variational Autoencoder (GeneVAE)
 ![GeneVAE](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-/blob/master/GeneVAE.png)

### Molecular Variational Autoencoder(JTVAE)
![JTVAE](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-/blob/master/JTVAE.png)

## Contents
* [Datasets](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-#Datasets)
* [Models](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-#Models)
* [Papers](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-#Papers)
* [Simulations](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-#Simulations)
* [Results](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-#Results)
 

### Datasets
1.  [CCLE_gene_expression_tpm](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-/tree/master/Dataset/CCLE)
1.  [TCGA_gene_expression_tpm](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-/tree/master/Dataset/TCGA)
2.  [Drug expression data SMILES](https://github.com/JIAQING-XIE/Machine-Learning-in-Genomes-Summer-Research-/tree/master/Dataset)
### Models
1. Variational Autoencoder(VAE)
2. Multilayer Perceptron(MLP)
