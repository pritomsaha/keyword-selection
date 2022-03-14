
import math
import numpy as np
import random
from gensim.models import KeyedVectors

def additive_smoothing(p, total_frequency):
    p = p*total_frequency + 1
    p /= (total_frequency + p.shape[0])
    return p


def get_mi_(eid1, eid2, eid2DocProb, eidDocPair2Prob):
    '''calculate mutual information between eid1 and eid2
    '''
    pair = frozenset([eid1, eid2])
    e1_prob1 = eid2DocProb.get(eid1, 0)
    e2_prob1 = eid2DocProb.get(eid2, 0)
    if eid1 == eid2:
        pair_prob_11 = e1_prob1

    else: pair_prob_11 = eidDocPair2Prob.get(pair,0)
    
    if pair_prob_11 == 0:
        return 0
    
    pmi = pair_prob_11*np.log(pair_prob_11/(e1_prob1*e2_prob1))/-np.log(pair_prob_11)
    return pmi

def get_npmi(eid1, eid2, eid2DocProb, eidDocPair2Prob):
    '''calculate normalized pointwise mutual information between eid1 and eid2
    '''
    pair = frozenset([eid1, eid2])
    e1_prob1 = eid2DocProb.get(eid1, 0)
    e2_prob1 = eid2DocProb.get(eid2, 0)
    if eid1 == eid2:
        pair_prob_11 = e1_prob1

    else: pair_prob_11 = eidDocPair2Prob.get(pair,0)
    
    if pair_prob_11 == 0:
        return -1
    
    pmi = math.log(pair_prob_11/(e1_prob1*e2_prob1))
    npmi = pmi/-math.log(pair_prob_11)
    return npmi

def get_mi(eid1, eid2, eid2DocProb, eidDocPair2Prob):
    '''calculate mutual information between eid1 and eid2
    '''
    mi = 0
    pair = frozenset([eid1, eid2])
    e1_prob1 = eid2DocProb.get(eid1, 0)
    e2_prob1 = eid2DocProb.get(eid2, 0)
    if eid1 == eid2:
        pair_prob_11 = e1_prob1

    else: pair_prob_11 = eidDocPair2Prob.get(pair,0)
    mi += 0 if pair_prob_11 == 0 else pair_prob_11*math.log(pair_prob_11/(e1_prob1*e2_prob1))
    
    e2_prob0 = 1 - e2_prob1
    pair_prob_10 = e1_prob1 - pair_prob_11
    mi += 0 if pair_prob_10 == 0 else pair_prob_10*math.log(pair_prob_10/(e1_prob1*e2_prob0))
    
    e1_prob0 = 1 - e1_prob1
    pair_prob_01 = e2_prob1 - pair_prob_11
    mi += 0 if pair_prob_01 == 0 else pair_prob_01*math.log(pair_prob_01/(e1_prob0*e2_prob1))
        
    pair_prob_00 = 1 - pair_prob_10 - pair_prob_01 - pair_prob_11    
    mi += 0 if pair_prob_00 == 0 else pair_prob_00*math.log(pair_prob_00/(e1_prob0*e2_prob0))
    
    return max(mi, 0)


def get_nmi(eid1, eid2, eid2DocProb, eidDocPair2Prob):
    '''calculate normalized mutual information between eid1 and eid2
    '''
    pair = frozenset([eid1, eid2])
    e1_prob1 = eid2DocProb.get(eid1, 0)
    e2_prob1 = eid2DocProb.get(eid2, 0)
    mi = 0
    norm = 0
    if eid1 == eid2:
        pair_prob_11 = e1_prob1
    else: pair_prob_11 = eidDocPair2Prob.get(pair,0)
    if pair_prob_11:
        mi += pair_prob_11*math.log(pair_prob_11/(e1_prob1*e2_prob1))
        norm += pair_prob_11*math.log(pair_prob_11)
   
    e2_prob0 = 1 - e2_prob1
    pair_prob_10 = e1_prob1 - pair_prob_11
    if pair_prob_10: 
        mi += pair_prob_10*math.log(pair_prob_10/(e1_prob1*e2_prob0))
        norm += pair_prob_10*math.log(pair_prob_10)
   
    e1_prob0 = 1 - e1_prob1
    pair_prob_01 = e2_prob1 - pair_prob_11
    if pair_prob_01:
        mi += pair_prob_01*math.log(pair_prob_01/(e1_prob0*e2_prob1))
        norm += pair_prob_01*math.log(pair_prob_01)
    
    pair_prob_00 = 1 - pair_prob_10 - pair_prob_01 - pair_prob_11
    if pair_prob_00:
        mi += pair_prob_00*math.log(pair_prob_00/(e1_prob0*e2_prob0))
        norm += pair_prob_00*math.log(pair_prob_00)
        
    nmi = min(mi/-norm, 1.0)
    nmi = max(0, nmi)
    return nmi


# load term candidates
def get_phrases(path):
    phrase_id = {}
    phrases = []
    with open(path) as fr:
        tid = 0
        for line in fr.readlines():
            w,v = line.split('\t')
            phrase_id[w] = tid
            phrases.append(w)
            tid += 1
    return phrase_id, phrases

# load word2vec embeddings
def load_embeddings(path, phrases):
    wv = KeyedVectors.load(path)
    X = []
    dim = len(wv[phrases[0].replace(' ', '_')])
    
    for phrase in phrases:
        phrase = phrase.replace(' ', '_')
        if phrase in wv:
            X.append(wv[phrase])
        else:
            X.append(np.random.rand(dim))

    return np.array(X)

# load GloVe embeddings
def load_embeddings_glove(path, phrases, phrase_connector=' '):
    gloveModel = {}
    with open(path) as f:
        for line in f:
            line = line.split()
            word = line[0]
            emb = np.array([float(v) for v in line[1:]])
            gloveModel[word] = emb
            
    dim = len(emb)
    X = []
    for phrase in phrases:
        ws = phrase.split(phrase_connector)
        emb = np.zeros(dim)
        
        for w in ws:
            if w in gloveModel:
                emb += gloveModel[w]
        X.append(emb)
    return np.array(X)