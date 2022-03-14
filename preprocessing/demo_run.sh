# an example preprocessing done on arxiv cs corpus

cd corpusProcess;
./corpusProcess/process.sh arxiv/cs/all -1 30;

cd ../featureExtraction;
./extract.sh arxiv/cs/all 30;
