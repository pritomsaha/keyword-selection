import csv
import os
import sys
from multiprocessing import Pool
from flashtext import KeywordProcessor
import pandas as pd
import re
from nltk.corpus import stopwords
stops = stopwords.words('english')

def segmentation(suffix):
    corpus_name = suffix[0]
    real_suffix = suffix[1]
    kp = suffix[2]

    input_path = "../../data/{}/intermediate/subcorpus-{}".format(corpus_name, real_suffix)
    output_path = "../../data/{}/intermediate/segmentation.txt-{}".format(corpus_name, real_suffix)
    
    text = None
    with open(input_path, 'r') as f:
        text = f.read()
    seg_text = kp.replace_keywords(text)
    
    with open(output_path, 'w') as f:
        f.write(seg_text)

def get_filtered(autophrase_output):
    filtered_phrases = []
    for entry in autophrase_output:
        phrase = entry.strip()
        if len(phrase)<2 or re.match("^[A-Za-z0-9\- ]*$", phrase) is None or phrase in stops:
            continue
        filtered_phrases.append(phrase)
    return filtered_phrases

if __name__ == '__main__':
    # python3 segmentation.py $DATA $THRESHOLD $THREAD
    corpus_name = sys.argv[1]
    number_of_processes = int(sys.argv[2])
    vocab_name = sys.argv[3]
    vocab_path = '../../data/vocabs/{}.txt'.format(vocab_name)
    
    candidate_keywords = pd.read_csv(vocab_path, sep='\t', header=None, keep_default_na=False, quoting=csv.QUOTE_NONE).values[:, 0]
    candidate_keywords = get_filtered(candidate_keywords)
    
    keyword_processor = KeywordProcessor()
    for keyword in candidate_keywords:
        keyword_processor.add_keyword(keyword, "<phrase>"+str(keyword)+"</phrase>")
    
    suffix_list = []
    for fileName in os.listdir('../../data/{}/intermediate/'.format(corpus_name)):
        if fileName.startswith("subcorpus-"):
            suffix_list.append((corpus_name, fileName[len("subcorpus-"):], keyword_processor))
            
    p = Pool(number_of_processes)
    p.map(segmentation, suffix_list)

