import pandas as pd
import numpy as np
import math
import csv
from em import EM
from dataLoader import *
from utils import *
from optimization import *
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import argparse
import yaml
import os

eng_stops = stopwords.words('english')

def getlog_odds(ywi, yw):
    ''' Log-odds on candidate keywords
    '''
    ni = np.sum(ywi)
    n = np.sum(yw)
    wwi = ywi/(ni - ywi)
    ww = yw/(n - yw)
    sigma_wi = np.log(wwi) - np.log(ww)
    var_wi = 1/ywi + 1/yw
    log_odds = sigma_wi/np.sqrt(var_wi)
    return log_odds

def keywords_PR(eids, eid2DocProb, eidDocPair2Prob):
    ''' PageRank on candidate keywords
    '''
    min_w = 0.0
    G = nx.Graph()
    edges = []
    for entry in eidDocPair2Prob.items():
        eid1, eid2 = list(entry[0])[0], list(entry[0])[1]
        w = get_mi(eid1, eid2, eid2DocProb, eidDocPair2Prob)
        if w >= min_w and eid1 in eids and eid2 in eids:
            edges.append((eid1, eid2, w))

    G.add_weighted_edges_from(edges)

    pr = nx.pagerank(G, alpha=0.9)
    pr = dict(sorted(pr.items(), key=lambda item: item[1], reverse=True))
    return pr

def get_tf(domain_path, bg_path, min_freq):
    domain_tf = pd.read_csv(domain_path + 'intermediate/entity2freq.txt', sep='\t',\
                            header=None, keep_default_na=False, quoting=csv.QUOTE_NONE).values
    bg_tf = pd.read_csv(bg_path + 'intermediate/entity2freq.txt', sep='\t',\
                        header=None, keep_default_na=False, quoting=csv.QUOTE_NONE).values
    dd = []
    for i, term in enumerate(domain_tf[:,0]):
        words = term.split("_")
        if words[0] in eng_stops or words[-1] in eng_stops: #filter out terms starting or ending with stopword
            continue
        if domain_tf[i,1] >= min_freq: #filter out terms with frequency less than min_freq 
              dd += [list(domain_tf[i])]
   
    domain_tf = np.array(dd)         
    domain_tf = domain_tf[np.argsort(domain_tf[:,0])]
    
    temp = bg_tf[np.in1d(bg_tf[:,0], domain_tf[:,0])]
    bg_tf = domain_tf.copy()
    bg_tf[np.in1d(domain_tf[:,0], temp[:,0])] = temp
    bg_tf = bg_tf[np.argsort(bg_tf[:,0])]
    del(temp)
    
    return domain_tf, bg_tf

def get_mi_mat(eids, eid2DocProb, eidDocPair2Prob):
    '''
    return term-by-term mutual information matrix
    '''
    N = eids.shape[0]
    mi_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i, N):
            mi_mat[i,j] = mi_mat[j,i] = get_mi(eids[i], eids[j], eid2DocProb, eidDocPair2Prob)
        
    return mi_mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/arxivcs.yaml",\
                        help="Which configuration to use. See into 'configs' folder")
    opt = parser.parse_args()
    with open(opt.config, 'r') as ymlfile:
         config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    domain_path = config['dataset']['domain_path']
    bg_path = config['dataset']['bg_path']
    result_path = os.path.join(config['dataset']['domain_path'], config['dataset']['result_folder'])
    min_freq = config['dataset']['min_freq']

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    eid2ename, ename2eid = loadEidToEntityMap(domain_path + 'intermediate/entity2id.txt')
    eid2DocProb = loadEid2DocFeature(domain_path + 'intermediate/eid2DocProb.txt')
    eidDocPair2Prob = loadEidDocPairFeature(domain_path + 'intermediate/eidDocPair2prob.txt')
    
    domain_tf, bg_tf = get_tf(domain_path, bg_path, min_freq)

    eids = np.array([ename2eid[ename] for ename in domain_tf[:,0]])
    P = domain_tf[:,1].astype('float')/np.sum(domain_tf[:,1].astype('float')) # term prob. dist. in domain corpus
    P_bg = bg_tf[:,1].astype('float')/np.sum(bg_tf[:,1].astype('float')) # term prob. dist. in background/context corpus

    C_w = domain_tf[:,1].astype('float')
    em_model = EM()
    llh, P_mm = em_model.run_model(C_w, P_bg.copy(), sigma=config['mixture_model']['sigma'],\
                        n_start=config['mixture_model']['n_start'], max_iter=config['mixture_model']['max_iter'],\
                        epsilon=float(config['mixture_model']['epsilon'])) # P_mm : term prob. dist. estimated from mixture model

    # additive smoothing P_mm to avoid zero term prob.
    total_frequency = np.sum(C_w)
    P_mm = additive_smoothing(P_mm, total_frequency)

    mi_mat = get_mi_mat(eids, eid2DocProb, eidDocPair2Prob)
    p_mi_mat = mi_mat/np.sum(mi_mat, axis=1, keepdims=True) # p_mi in eq.5 of the paper.

    K = eids.shape[0]
    selected_terms_rf = domain_tf[np.argsort(-P.astype(float))][:K,0]
    selected_terms_mm = domain_tf[np.argsort(-P_mm)][:K,0]

    selected_terms_kl_rf = np.array([eid2ename[eid] for eid in lazy_greedyPKS(K, p_mi_mat, eids,\
                                                                                P, P_bg, total_frequency)])[:K]
    selected_terms_kl_mm = np.array([eid2ename[eid] for eid in lazy_greedyPKS(K, p_mi_mat, eids,\
                                                                                   P_mm, P_bg, total_frequency)])[:K]
    selected_terms_FL = np.array([eid2ename[eid] for eid in lazy_greedyFL(K, mi_mat, eids)])[:K]

    log_odds = getlog_odds(domain_tf[:,1].astype(int), bg_tf[:,1].astype(int))
    selected_terms_lo = domain_tf[np.argsort(-log_odds)][:K,0]

    pr = keywords_PR(eids, eid2DocProb, eidDocPair2Prob)
    selected_terms_pr = np.array([eid2ename[eid] for eid in list(pr.keys())])[:K]
    
    # save the selected terms by baselines and proposed models in results folder of the dataset
    np.savetxt(result_path+'rf.txt', selected_terms_rf,fmt='%s')
    np.savetxt(result_path+'lo.txt', selected_terms_lo,fmt='%s')
    np.savetxt(result_path+'fl.txt', selected_terms_FL,fmt='%s')
    np.savetxt(result_path+'pr.txt', selected_terms_pr,fmt='%s')
    np.savetxt(result_path+'kl_rf.txt', selected_terms_kl_rf,fmt='%s')
    np.savetxt(result_path+'mm.txt', selected_terms_mm,fmt='%s')
    np.savetxt(result_path+'kl_mm.txt', selected_terms_kl_mm,fmt='%s')