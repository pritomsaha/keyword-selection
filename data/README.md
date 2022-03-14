# Domain Representative Keywords Selection: A Probabilistic Approach


## Introduction

This is the source code for keyword selection framework developed for selecting a set of k keywords from a candidate set.


## Usage

We provide the data preprocessing code, and the python implementation of our method and baselines specified in the paper. 

If you want to use our data preprocessing code, then you need to create a folder with your dataset name under "/.data/" folder and put the corpus on the "source" folder. How to run the preporcessing is shown in preprocessing folder.

Otherwise, you can directly download our preprocessed datasets and other groundtruth data used in the experiments from [Google Drive](https://drive.google.com/drive/folders/1ePRLRblSUlA8jHTNpiaNNbBXQp15GJmW?usp=sharing); unzip it and put the dataset in under the "./data/" folder.


## To Run

```
cd ./src
python3 keyword_selection.py --config config_filename
```
Please see the some config files in the "./src/configs/" folder that are used for the experients presented in the paper. 
Results are saved under the results folder of the corresponding dataset folder"

## Publications

Please cite the following paper if you are using this code. Thanks!

* Pritom Saha Akash, Jie Huang, Kevin Chen-Chuan Chang, Yunyao Li, Lucian Popa, ChengXiang Zhai. Domain Representative Keywords Selection: A Probabilistic Approach. Findings of the 60th Annual Meeting of the Association for Computational Linguistics (Findings of ACL). 2022
