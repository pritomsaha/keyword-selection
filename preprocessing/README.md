The proprocessing module is adapted from [SetExpan](https://github.com/jmshen1994/SetExpan/tree/master/src).


## Steps to follow for preprocessing

- First create a folder for your dataset and a subfolder with corpus file named "/source".

- To process the corpus:
```
cd corpusProcess;
./process.sh dataset_folder_name vocab_file NUM_THREAD;
```
- If you want to automatically generate vocabulary using AutoPhrase then set -1 for "vocab_file" in the command. Or give the vocab file name and put the file in the "./data/vocabs/" folder. 


- To generate features from processed corpus:
```
cd featureExtraction;
./extract.sh dataset_folder_name NUM_THREAD;
```
