# Opcode N-gram

---
>ref. Li, Xiang, et al. "An adversarial machine learning method based on OpCode N-grams feature in malware detection." 2020 IEEE Fifth International Conference on Data Science in Cyberspace (DSC). IEEE, 2020.

---

This project is based on [Opcode_detector](https://github.com/Jim16888/Opcode_detector).

## Introduction

### Descripition

* This is a malware detector which use the opcode as feature
    
    - Input:a binary file
    
    - output:the probability of each class which is predicted by Machine method
    
### Feature Extraction

* feature 
    - opcode sequence

* extracted tool
    
     - radare2 4.3.1

## Requirement
---
* python3
* radare2
* python package
    - r2pipe
    - sklearn
    - joblib
    - xgboost

## File
* FC_model : the models are responsible for classify the malware family (saved by joblib)
* MD_model : the models are responsible for classify the malware or benignware (saved by joblib)
* TestingBin : the file that can test this detector
* main.py : the detector
* utils.py : for parsing args
* top_feature_1.npy : save the training data about vectorize the opcode sequence

## Usage
* input binary: `-i <path>`, `--input-path <path>`
* model: `-m <model>`, `--model <model>`
  * xgboost, SVM
* output (record): `-o <path>`, `--output-path <path>`
* Malware Detection / Family Classification
    * do nothing if you wanna do malware detection(binary clf)  
    * add `-c` if you wanna do family classification 
* e.g.
    `python main.py -i TestingBin/malware/00a2bd396600e892da75c686be60d5d13e5a34824afd76c02e8cf7257d2cf5c5 -o myDetector_FC_records.csv -m xgboost -c`
    * using trained rf family classifier(`-c`), predict '00a2bd396600e892da75c686be60d5d13e5a34824afd76c02e8cf7257d2cf5c5' and write the result to 'myDetector_FC_records.csv'
    * add `-W ignore` if you keep getting bothered by warning msg

MODEL can be xgboost, SVM  default: xgboost

* output file format

  |    Filename  | Benignware | Malware |
  | :----------: | :------: | :-------: |
  | 00ffe391     |   0.97   |   0.03    |
  |     00f391fe      |  0.967   |  0.033   |
  |     1fe00f39      |  -1   |    |
  * it will record the prob of each class
  * -1 means fail

### Accuracy
* classification
    * SVM : 0.8753
    * xgboost : 0.9808
* Detect
    * SVM : 0.963
    * xgboost : 0.9963

### Hackmd
> https://hackmd.io/@rkZ7JhsGQ7OWFewsWSx7KQ/H1GHHKGEo
