from utils import *
import numpy as np
from priorityQueue import PriorityQueue as PQ

def relative_entropy(p, q):
    # smoothing to avoid divide by zero
    q = additive_smoothing(q, total_frequency)
    div = np.sum(p*np.log(p/q))
    return div

def get_gainPKS(i, mat, qt, p, p_bg):
    ''' this function calculate gain for the proposed prob. keyword selection (PKS)
    '''
    q = qt + mat[i,:]*p[i]
    return relative_entropy(p, qt) - relative_entropy(p, q)

def greedyPKS(K, mat, vocab, p, p_bg, total_freq):
    ''' greedy version of PKS
    '''
    global total_frequency
    total_frequency = total_freq

    Q = np.zeros(vocab.shape)
    S = []
    idx = []
    for _ in range(K):
        max_gain = -np.inf
        best_i = None
        for i in range(vocab.shape[0]):
            if i in idx: continue
            g = get_gainPKS(i, mat, Q, p, p_bg)
            if g > max_gain:
                max_gain = g
                best_i = i
        Q += mat[best_i,:]*p[best_i]
        idx += [best_i]
        S += [vocab[best_i]]
    return S

def lazy_greedyPKS(K, mat, vocab, p, p_bg, total_freq):
    ''' lazy greedy version of PKS
    '''
    global total_frequency
    total_frequency = total_freq
    
    Q = np.zeros(vocab.shape)
    gains = PQ()
    S = []
    for i in range(vocab.shape[0]):
        g = get_gainPKS(i, mat, Q, p, p_bg)
        gains.add_task(i, -g)

    for _ in range(K):
        i, g = gains.pop_item()
        while True:
            gain = -get_gainPKS(i, mat, Q, p, p_bg)
            gains.add_task(i, gain)
            j, g = gains.pop_item()
            if gain > g:
                i = j
                continue
            
            S += [vocab[j]]
            Q += mat[j,:]*p[j]
            break    
    return S

def get_gainFL(i, mat, qt):
    ''' this function calculate gain for Facility Location (FL) baseline
    '''
    max_idx = qt < mat[i,:]
    q = qt.copy()
    q[max_idx] = mat[i,max_idx]

    return np.sum(q) - np.sum(qt)


def greedyFL(K, mat, vocab):
    ''' greedy version of FL
    '''
    # initializationn
    Q = np.zeros(vocab.shape)
    S = []
    idx = []
    for _ in range(K):
        max_gain = -np.inf
        best_i = None
        for i in range(vocab.shape[0]):
            if i in idx: continue
            g = get_gainFL(i, mat, Q)
            if g > max_gain:
                max_gain = g
                best_i = i
        max_idx = Q < mat[best_i,:]
        Q[max_idx] = mat[best_i,max_idx]
        idx += [best_i]
        S += [vocab[best_i]]
  
    return S

def lazy_greedyFL(K, mat, vocab):
    ''' lazy greedy version of FL
    '''
    # initializationn
    Q = np.zeros(vocab.shape)
    gains = PQ()
    S = []
    idx = []
    for i in range(vocab.shape[0]):
        g = get_gainFL(i, mat, Q)
        gains.add_task(i, -g)

    for _ in range(K):
        i, g = gains.pop_item()
        while True:
            gain = -get_gainFL(i, mat, Q)
            gains.add_task(i, gain)
            j, g = gains.pop_item()
            if gain > g:
                i = j
                continue
            
            idx += [j]
            S += [vocab[j]]
            max_idx = Q < mat[j,:]
            Q[max_idx] = mat[j,max_idx]
            break     
    return S