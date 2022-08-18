import sys
sys.path.append('../')
import torch
import torch.nn as nn
from optparse import OptionParser
from tqdm import tqdm
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import numpy as np  
from fast_jtnn import *
from fast_jtnn import sascorer
import networkx as nx
import os
import pandas as pd


def scorer(smiles):
    smiles_rdkit = []
    for i in range(len(smiles)):
        smiles_rdkit.append(
            MolToSmiles(MolFromSmiles(smiles[i]), isomericSmiles=True))

    logP_values = []
    for i in range(len(smiles)):
        logP_values.append(
            Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[i])))

    SA_scores = []
    for i in range(len(smiles)):
        SA_scores.append(
            -sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i])))

    cycle_scores = []
    for i in range(len(smiles)):
        cycle_list = nx.cycle_basis(
            nx.Graph(
                rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[i]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)

    SA_scores_normalized = (
        np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (
        np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (
        np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    targets = (SA_scores_normalized +
               logP_values_normalized +
               cycle_scores_normalized)

    return (SA_scores,
            logP_values,
            cycle_scores,
            targets)


def main_gen_latent(data_path, vocab_path,
                    model_path, output_path='./',
                    hidden_size=450, latent_size=56,
                    depthT=20, depthG=3, batch_size=1):
    with open(data_path) as f:
        smiles = f.readlines()
    
    if os.path.isdir(output_path) is False:
        os.makedirs(output_path)

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()

    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    drug_id_filepath = os.path.join("../data", 'DRUG_SMILES.txt')
    df = pd.read_table(open(drug_id_filepath), sep='\t')

    tot = 0
    model.eval()
    with torch.no_grad():
        temp = ['d'+str(k) for k in range(0, 56)]
        temp.insert(0, 'DRUG_ID')
        latent_points = [temp]
        for i in tqdm(range(0, len(smiles), batch_size)):
            batch = smiles[i:i + batch_size]
            try:
                mol_vec = model.encode_from_smiles(batch)
                npVec = mol_vec.data.cpu().numpy()[0]
                listVec = npVec.tolist()
                listVec.insert(0, df.loc[i]['DRUG_ID'])
                latent_points.append(torch.Tensor(listVec))
                tot += 1
            except:
                latent_points.append(np.array([0 for i in range(0, 57)]))


    latent_points = np.vstack(latent_points)

    SA_scores, logP_values, cycle_scores, targets = scorer(smiles)
    np.savetxt(
        os.path.join(output_path, 'latent_features.txt'), latent_points, fmt = '%s')
    np.savetxt(
        os.path.join(output_path, 'targets.txt'), targets)
    np.savetxt(
        os.path.join(output_path, 'logP_values.txt'), np.array(logP_values))
    np.savetxt(
        os.path.join(output_path, 'SA_scores.txt'), np.array(SA_scores))
    np.savetxt(
        os.path.join(output_path, 'cycle_scores.txt'), np.array(cycle_scores))


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-a", "--data", dest="data_path", default='../data/smiles.txt')
    parser.add_option("-v", "--vocab", dest="vocab_path", default='../data/vocab.txt')
    parser.add_option("-m", "--model", dest="model_path", default='../fast_molvae/vae_model/model.epoch-19')
    parser.add_option("-o", "--output", dest="output_path", default='../data/')
    parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
    parser.add_option("-l", "--latent", dest="latent_size", default=56)
    parser.add_option("-t", "--depthT", dest="depthT", default=20)
    parser.add_option("-g", "--depthG", dest="depthG", default=3)

    opts, args = parser.parse_args()

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depthT = int(opts.depthT)
    depthG = int(opts.depthG)

    main_gen_latent(opts.data_path, opts.vocab_path,
                    opts.model_path, output_path=opts.output_path,
                    hidden_size=hidden_size, latent_size=latent_size,
                    depthT=depthT, depthG=depthG)
