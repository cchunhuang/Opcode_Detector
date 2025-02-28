# Opcode Malware Detector

## Introduction
This project is based on research from the paper: Li, Xiang, et al. "An adversarial machine learning method based on OpCode N-grams feature in malware detection." 2020 IEEE Fifth International Conference on Data Science in Cyberspace (DSC). IEEE, 2020.

The paper introduces a method for adversarial machine learning in malware detection using OpCode N-grams as features. By leveraging TF-IDF to extract OpCode sequences and applying models such as XGBoost, the research demonstrates how adversarial features can be used to fool machine learning-based malware detectors. This project builds upon those insights to enhance malware detection capabilities.

This project is a **Malware Detection System** using **Support Vector Machine (SVM) and XGBoost**. It extracts features from executable files, vectorizes them, and trains models to detect malware.

This project partially utilizes the work from [Opcode_detector](https://github.com/Jim16888/Opcode_detector).

## Project Structure
```
├── src/
│   ├── code/
│   │   ├── main.py                 # Main script to run training and prediction
│   │   ├── MalwareDetector.py      # Malware detection model class
│   │   ├── Logger.py               # Logger setup utility
│   │   ├── utils.py                # Feature extraction and vectorization functions
│   ├── config/
│   │   ├── config_SVM.json         # Configuration for SVM model
│   │   ├── config_XGBoost.json     # Configuration for XGBoost model
│   │   ├── logging_config.json     # Logging configuration
│   ├── resource/
│   │   ├── top_features_1.npy        # Feature extraction data
```

## Features
- **Feature Extraction:** Extracts opcode sequences using `r2pipe`.
- **Vectorization:** Converts opcode sequences into numerical vectors.
- **Machine Learning Models:** Supports **SVM** and **XGBoost**.
- **Logging System:** Tracks execution and model performance.
- **Configuration Driven:** Uses JSON config files for easy setup.

## Installation
### Prerequisites
This project was developed using Python 3.11.5. To ensure compatibility, please use this version or later.

The required dependencies are listed in requirements.txt. You can install them using:
  ```sh
pip install -r requirements.txt
  ```

## Usage
For details on configuring the project, please refer to the provided documentation within the project.
### SVM Model:
```sh
python main.py config_SVM.json
```
### XGBoost Model:
```sh
python main.py config_XGBoost.json
```

