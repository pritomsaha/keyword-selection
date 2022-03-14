import sys
import time
import json
import re
from tqdm import tqdm
from collections import deque
import spacy
from spacy.symbols import ORTH, LEMMA, POS, TAG
import mmap
import argparse

DEBUG = True

# INIT SpaCy
nlp = spacy.load('en_core_web_lg')
# start_phrase = [{ORTH: u'<phrase>', LEMMA: u'', POS: u'START_PHRASE', TAG: u'START_PHRASE'}]
# end_phrase = [{ORTH: u'</phrase>', LEMMA: u'', POS: u'END_PHRASE', TAG: u'END_PHRASE'}]

nlp.get_pipe("attribute_ruler").add([[{"TEXT": "<phrase>"}]], {"POS": "START_PHRASE"}, {"LEMMA": "START_PHRASE"})
nlp.get_pipe("attribute_ruler").add([[{"TEXT": "</phrase>"}]], {"POS": "END_PHRASE"}, {"LEMMA": "END_PHRASE"})
# nlp.tokenizer.add_special_case(u'<phrase>', start_phrase)
# nlp.tokenizer.add_special_case(u'</phrase>', end_phrase)

p2tok_list = {}  # global cache of phrase to token


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # add space before and after <phrase> tags
    text = re.sub(r"<phrase>", " <phrase> ", text)
    text = re.sub(r"</phrase>", " </phrase> ", text)
    # text = re.sub(r"<phrase>", " ", text)
    # text = re.sub(r"</phrase>", " ", text)
    # add space before and after special characters
    text = re.sub(r"([.,!:?()])", r" \1 ", text)
    text = " ".join(text.split("-"))
    text = " ".join(text.split())
    # replace multiple continuous whitespace with a single one
    text = re.sub(r"\s{2,}", " ", text)

    return text


def find(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.

    >>> find([1, 1, 2], [1, 2])
    1

    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j].lower() != needle[-j - 1]:
                i += skip.get(haystack[i].lower(), n)
                break
        else:
            return i - n + 1
    return -1


def obtain_p_tokens(p):
    '''
    :param p: a phrase string
    :return: a list of token text
    '''

    if p in p2tok_list:
        return p2tok_list[p]
    else:
        p_tokens = [tok.text for tok in nlp(p)]
        p2tok_list[p] = p_tokens
        return p_tokens



def process_one_doc(article, articleId):
    result = []
    phrases = []
    output_token_list = []
    
    # go over once
    article = clean_text(article).strip()
    q = deque()
    IN_PHRASE_FLAG = False
    for token in article.split():
        if token == "<phrase>":
            IN_PHRASE_FLAG = True
        elif token == "</phrase>":
            current_phrase_list = []
            while (len(q) != 0):
                current_phrase_list.append(q.popleft())
            phrases.append(" ".join(current_phrase_list).lower())
            IN_PHRASE_FLAG = False
        else:
            if IN_PHRASE_FLAG:  # in the middle of a phrase, push the token into queue
                q.append(token)

            ## put all the token information into the output fields
            output_token_list.append(token)

    text = " ".join(output_token_list)
    doc = nlp(text)
    j = 0
    sentId = 0
    for i, sent in enumerate(doc.sents):  # seems to me doc.sents is just to separate a sentence into several parts (according to ':')
        NPs = []
        pos = []

        lemmas = []
        deps = []

        tokens = []
        for s in sent.noun_chunks:
            NPs.append(s)

        nn = []
        # get pos tag and dependencies
        for token in sent:
            tokens.append(token.text)
            pos.append(token.tag_)
            lemmas.append(token.lemma_)
            deps.append(token.dep_)
            if token.pos_ == "NOUN":
                nn.append(token.text)

        entityMentions = []
        offset = 0
        used_phrase = 0
        for p in phrases[j:]:
            if re.match("^[A-Z0-9a-z ]*$", p) is None or len(p.strip())<2 : continue
            if len(p.split()) == 1 and p not in nn: continue
            is_NP = False
            rest_sent = " ".join(tokens[offset:])
            for k, np in enumerate(NPs):
                p_sidx = rest_sent.lower().find(p)
                if p_sidx == -1: continue
                p_eidx = p_sidx + len(p) - 1
                np_sidx = rest_sent.lower().find(np.text)
                if np_sidx == -1: continue
                np_eidx = np_sidx + len(np.text) - 1
                if p_sidx >= np_sidx and p_eidx <= np_eidx:
                    is_NP = True
                    break
                
            p_tokens = obtain_p_tokens(p)
            idx = find(tokens[offset:], p_tokens)
            
            if idx == -1: continue
            
            start = offset + idx
            end = start + len(p_tokens) - 1
            offset = end + 1
            used_phrase += 1
            
            if is_NP is False: continue
            
            ent = {"text": " ".join(p_tokens), "start": start,
                       "end": end, "type": "phrase"}
            
            entityMentions.append(ent)
        
        j += used_phrase
        res = {"articleId": articleId, "sentId": sentId, "tokens": tokens, "pos": pos, "lemma": lemmas, "dep": deps,
               "entityMentions": entityMentions,
               "np_chunks": [{"text": t.text, "start": t.start - sent.start, "end": t.end - sent.start - 1} for t in
                             NPs]}
        result.append(res)
        sentId += 1
    
    return result


def process_corpus(input_path, output_path, real_suffix):
    start = time.time()
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for cnt, line in tqdm(enumerate(fin), total=get_num_lines(input_path)):
            line = line.strip()
            # try:
            article_result = process_one_doc(line, "{}-{}".format(real_suffix, cnt))
            for sent in article_result:
                json.dump(sent, fout)
                fout.write("\n")
            # except:
            #     print("exception")
    end = time.time()
    print("Finish NLP processing, using time %s (second)" % (end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main.py', description='')
    parser.add_argument('-corpusName', required=False, default='sample_dataset', help='corpusName: sample_dataset or sample_wiki or wiki')
    parser.add_argument('-input_path', required=False, help='input_path')
    parser.add_argument('-output_path', required=False, help='output_path')
    parser.add_argument('-real_suffix', required=False, help='real_suffix: used to prepend for articleID')  # used to prepend for articleID
    args = parser.parse_args()
    process_corpus(args.input_path, args.output_path, args.real_suffix)
