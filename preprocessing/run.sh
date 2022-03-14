# if you have external vocabulary then 

cd corpusProcess; 
./process.sh arxiv/cs/all -1 30;
cd ../featureExtraction;
./extract.sh arxiv/cs/all 30;

cd ..
