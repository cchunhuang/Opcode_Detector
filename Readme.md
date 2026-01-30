# Opcode Malware Detector

## Introduction
This project is based on research from the paper: Li, Xiang, et al. "An adversarial machine learning method based on OpCode N-grams feature in malware detection." 2020 IEEE Fifth International Conference on Data Science in Cyberspace (DSC). IEEE, 2020.

The paper introduces a method for adversarial machine learning in malware detection using OpCode N-grams as features. By leveraging TF-IDF to extract OpCode sequences and applying models such as XGBoost, the research demonstrates how adversarial features can be used to fool machine learning-based malware detectors. This project builds upon those insights to enhance malware detection capabilities.

This project is a **Malware Detection System** using **Support Vector Machine (SVM) and XGBoost**. It extracts opcode sequences from executable files using radare2, vectorizes them using n-gram features, and trains machine learning models to detect malware.

This project partially utilizes the work from [Opcode_detector](https://github.com/Jim16888/Opcode_detector).

## Project Structure
```
Opcode_Detector/
├── src/
│   ├── main.py                     # Main script to run training and prediction
│   ├── MalwareDetector.py          # Malware detection model class
│   ├── utils.py                    # Feature extraction and vectorization functions
│   ├── config_SVM.json             # Configuration for SVM model
│   ├── config_XGBoost.json         # Configuration for XGBoost model
│   └── top_features_1.npy          # Pre-selected top n-gram features
├── output_SVM/                     # SVM model outputs
│   ├── log/                        # Training logs
│   ├── model/                      # Saved models and training results
│   └── predict/                    # Prediction results
├── output_XGBoost/                 # XGBoost model outputs
│   ├── log/                        # Training logs
│   ├── model/                      # Saved models and training results
│   └── predict/                    # Prediction results
├── requirements.txt
└── Readme.md
```

## Features
- **Feature Extraction:** Extracts opcode sequences from binary files using `r2pipe` and radare2
- **Parallel Processing:** Supports multi-core parallel extraction with configurable `n_jobs` parameter
- **N-gram Vectorization:** Converts opcode sequences into numerical vectors using 2-4 gram features
- **Machine Learning Models:** Supports **SVM** (linear kernel) and **XGBoost** classifiers
- **Flexible Workflow:** Supports both training and prediction modes
- **Model Persistence:** Save and load trained models for reuse
- **Comprehensive Logging:** Tracks execution, model performance, and predictions
- **Configuration Driven:** Fully configurable via JSON config files

## Installation

### Prerequisites
- **Python Version:** Python 3.11.5 or later
- **Radare2:** Required for binary analysis and opcode extraction
  ```sh
  # Install radare2 on Ubuntu/Debian
  sudo apt-get install radare2
  ```

### Dependencies
Install all required Python packages using `requirements.txt`:
```sh
pip install -r requirements.txt
```

## Dataset Format

The dataset label file (CSV) should follow this format:

```csv
filename,label,type
0000174b098ffbbab221cd21cc7d7c4217abbc923e223f80acff4dc7f3d2dfe3,malware,train
00006b107f074baad04c044c9a8800c97e55cb4df7406a0e2f954fed00741da2,benignware,train
0001905dbdb8fe27595f83df406f45fd26d2af856285a0725a43713f2489e6f1,malware,test
...
```

- **filename:** Name of the binary file
- **label:** Either `malware` or `benignware`
- **type:** Either `train`, `test`, or `predict`

## Configuration Guide

The system uses JSON configuration files to control all aspects of training and prediction. Two example configurations are provided:
- `config_SVM.json` - Configuration for Support Vector Machine
- `config_XGBoost.json` - Configuration for XGBoost

### Configuration Structure

#### 1. File Paths (`file`)
Specifies all input and output file paths:

```json
"file": {
    "label": "./dataset/label.csv",
    "top_features": "./src/top_features_1.npy",
    "input_model": "",
    "output_model": "./output_SVM/model/SVM.pkl",
    "train_result": "./output_SVM/model/score.json",
    "predict_result": "./output_SVM/predict/predict_result.json"
}
```

**Parameters:**
- `label` (string, required): Path to the CSV file containing file labels and train/test/predict splits
- `top_features` (string, required): Path to the `.npy` file containing pre-selected n-gram features for vectorization
- `input_model` (string, optional): Path to a pre-trained model file. Leave empty (`""`) to train from scratch
- `output_model` (string, required): Path where the trained model will be saved (`.pkl` format)
- `train_result` (string, optional): Path where training metrics (accuracy, precision, recall, F1) will be saved as JSON
- `predict_result` (string, optional): Path where prediction results will be saved as JSON

#### 2. Folder Paths (`folder`)
Specifies output directories for different components:

```json
"folder": {
    "log": "./output_SVM/log/",
    "dataset": "./dataset/",
    "feature": "./output_SVM/feature/",
    "vector": "./output_SVM/vector/",
    "model": "./output_SVM/model/",
    "predict": "./output_SVM/predict/"
}
```

**Parameters:**
- `log` (string): Directory for storing execution logs
- `dataset` (string): Directory containing the binary executable files
- `feature` (string): Directory for storing extracted features
- `vector` (string): Directory for storing vectorized data
- `model` (string): Directory for storing trained models
- `predict` (string): Directory for storing prediction results

#### 3. Parameters (`params`)

##### General Parameters
```json
"params": {
    "n_jobs": -1,
    "mode": "detection",
    ...
}
```

- `n_jobs` (integer): Number of parallel processes for feature extraction
  - `-1`: Use all available CPUs (recommended)
  - `1`: Sequential processing (useful for debugging)
  - `n > 1`: Use n parallel processes
- `mode` (string): Operation mode, currently only `"detection"` is supported

##### Model Parameters for SVM (`params.model`)
```json
"model": {
    "model_name": "SVM",
    "kernel_type": "linear",
    "probability": true,
    "test_size": 0.3,
    "random_state": 42
}
```

**SVM-specific parameters:**
- `model_name` (string): Must be `"SVM"`
- `kernel_type` (string): SVM kernel type
  - `"linear"`: Linear kernel (recommended for high-dimensional text features)
  - `"rbf"`: Radial basis function kernel
  - `"poly"`: Polynomial kernel
  - `"sigmoid"`: Sigmoid kernel
- `probability` (boolean): Enable probability estimates (required for confidence scores)
- `test_size` (float): Proportion of dataset used for testing (0.0 to 1.0)
- `random_state` (integer): Random seed for reproducibility

##### Model Parameters for XGBoost (`params.model`)
```json
"model": {
    "model_name": "XGBoost",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "use_label_encoder": false,
    "random_state": 42,
    "test_size": 0.3
}
```

**XGBoost-specific parameters:**
- `model_name` (string): Must be `"XGBoost"`
- `n_estimators` (integer): Number of boosting rounds (trees)
  - Default: `100`
  - Higher values may improve accuracy but increase training time
- `max_depth` (integer): Maximum tree depth
  - Default: `6`
  - Controls model complexity and overfitting
- `learning_rate` (float): Step size shrinkage to prevent overfitting
  - Default: `0.1`
  - Range: 0.0 to 1.0
  - Lower values require more `n_estimators`
- `eval_metric` (string): Evaluation metric for validation
  - `"logloss"`: Logarithmic loss for binary classification
  - `"error"`: Binary classification error rate
  - `"auc"`: Area under the curve
- `use_label_encoder` (boolean): Use XGBoost's built-in label encoder
  - Set to `false` (deprecated feature in newer XGBoost versions)
- `random_state` (integer): Random seed for reproducibility
- `test_size` (float): Proportion of dataset used for testing (0.0 to 1.0)

#### 4. Action (`action`)
```json
"action": "train"
```

Specifies the operation mode:
- `"train"`: Extract features, train model, evaluate on test set, and make predictions
- `"predict"`: Load existing model and make predictions only


## Usage

### Training a New Model

#### Train SVM Model:
```sh
cd src
python main.py config_SVM.json
```

#### Train XGBoost Model:
```sh
cd src
python main.py config_XGBoost.json
```

**Training Process:**
1. Loads dataset labels from CSV file
2. Extracts opcode sequences from binary files using radare2
3. Vectorizes opcode sequences using n-gram features (2-4 grams)
4. Trains the specified model on training data
5. Evaluates on test data and calculates metrics (accuracy, precision, recall, F1)
6. Saves the trained model and evaluation results
7. Makes predictions on files marked as `predict` type

### Using a Pre-trained Model for Prediction

To use an existing model for prediction only:

1. Modify your config file:
   - Set `"action": "predict"`
   - Set `"input_model"` to your trained model path
   - Update `"label"` to point to your prediction dataset

2. Run prediction:
```sh
cd src
python main.py your_config.json
```

### Output Files

#### Training Results (`train_result`)
Example: `output_SVM/model/score.json`
```json
{
    "final_result": {
        "TP": 150,
        "TN": 145,
        "FP": 5,
        "FN": 8,
        "accuracy": 0.9578,
        "precision": 0.9677,
        "recall": 0.9494,
        "f1_score": 0.9585
    }
}
```

#### Prediction Results (`predict_result`)
Example: `output_SVM/predict/predict_result.json`
```json
[
    {
        "name": "0000174b098ffbbab221cd21cc7d7c4217abbc923e223f80acff4dc7f3d2dfe3",
        "detection": "malware"
    },
    {
        "name": "00006b107f074baad04c044c9a8800c97e55cb4df7406a0e2f954fed00741da2",
        "detection": "benignware"
    }
]
```

## Performance Tips

1. **Parallel Processing:** Use `"n_jobs": -1` to utilize all CPU cores for faster feature extraction
2. **Model Selection:** 
   - SVM with linear kernel is fast and works well for high-dimensional sparse features
   - XGBoost may provide better accuracy but requires more training time
3. **Feature Engineering:** The `top_features_1.npy` file contains pre-selected important n-gram features. You can generate your own feature set for different datasets.

## References

- Li, Xiang, et al. "An adversarial machine learning method based on OpCode N-grams feature in malware detection." 2020 IEEE Fifth International Conference on Data Science in Cyberspace (DSC). IEEE, 2020.
- [Opcode_detector](https://github.com/Jim16888/Opcode_detector) - Original implementation reference

## License

See the [LICENSE](LICENSE) file for details.

