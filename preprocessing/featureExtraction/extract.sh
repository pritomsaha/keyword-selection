#!/bin/bash
DATA=$1
path=$(pwd)

## The embedding method used, currently support word2vec
EMBEDDING_METHOD=word2vec

## Number of threads used to learn word2vec embedding
EMBED_LEARNING_THREAD=$2

green=`tput setaf 2`
reset=`tput sgr0`
echo ${green}==='Corpus Name:' $DATA===${reset}
echo ${green}==='Current Path:' $path===${reset}


if [ ! -d ../../data/$DATA/intermediate ]; then
	mkdir ../../data/$DATA/intermediate
fi


echo ${green}==='Extract Document-level Co-occurrence Features'===${reset}
python3 extractEidDocPairFeature.py $DATA

echo ${green}==='Extract Embedding Features (using word2vec)'===${reset}
python3 learnEmbedFeature.py $DATA $EMBED_LEARNING_THREAD