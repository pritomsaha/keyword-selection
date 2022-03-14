"""
__author__: Jiaming Shen
__description__: extract entity pair document level co-occurrence features
    Input: 1) the sentence.json
    Output: 1) eidDocPairCounts.txt, 2) eidDocPairPPMI.txt
"""
import sys
import json
import itertools
import numpy as np
import math
from collections import defaultdict
import mmap
from tqdm import tqdm
import re


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python3 extractEidDocPairFeature.py -data')
        exit(1)
    corpusName = sys.argv[1]

    infilename = '../../data/' + corpusName + '/intermediate/sentences.json'
    outfilename1 = '../../data/' + corpusName + '/intermediate/eid2DocProb.txt'
    outfilename2 = '../../data/' + corpusName + '/intermediate/eidDocPair2prob.txt'

    articeID2eidlist = defaultdict(list)

    with open(infilename, 'r') as fin:
        for line in tqdm(fin, total=get_num_lines(infilename), desc="Generate document-level cooccurrence features (pass 1)"):
          sentInfo = json.loads(line)
          articleId = sentInfo['articleId']
          eidlist = [em['entityId'] for em in sentInfo['entityMentions']]
          articeID2eidlist[articleId].extend(eidlist)            

    docLen = len(articeID2eidlist.keys())
    eid2DocFreq = defaultdict(int)
    eidPair2count = defaultdict(int)

    for articleId in tqdm(articeID2eidlist, desc="Generate document-level coocurrence features (pass 2)"):
        eidlist = set(articeID2eidlist[articleId])
        for pair in itertools.combinations(eidlist,2):
            eidPair2count[frozenset(pair)] += 1
        for eid in eidlist:
            eid2DocFreq[eid] += 1

    with open(outfilename1, 'w') as fout:
        for eid in eid2DocFreq:
            prob = eid2DocFreq[eid]/docLen
            fout.write(str(eid) + "\t" + str(prob)+ "\n")            
            
    with open(outfilename2, 'w') as fout:
        for pair in eidPair2count:
            prob = eidPair2count[pair]/docLen
            pair = list(pair)
            fout.write(str(pair[0]) + "\t" + str(pair[1]) + "\t" + str(prob) + "\n")