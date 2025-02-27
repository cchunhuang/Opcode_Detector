# config_SVM.json

`config_SVM.json` is a configuration file for setting up the SVM (Support Vector Machine) malware detection model. This file includes settings for model training, prediction, and related paths and parameters.

## Configuration Structure

```json
{
    "config": {
        "path": {
            "input_file": null,
            "train_result": "./output/model/score.json",
            "predict_result": "./output/predict/predict_result.json",
            "input_model": "",
            "output_model": "./output/model/SVM.pkl",
            "top_features": "./src/resource/top_features_1.npy",
            "logging_config": "./src/config/logging_config.json",
            "log_file": "./output/log/logging.log"
        },
        "folder": {
            "output": "./output",
            "model": "./output/model",
            "predict": "./output/predict",
            "log": "./output/log"
        },
        "model": {
            "model_name": "SVM",
            "kernel_type": "linear",
            "probability": true,
            "test_size": 0.3,
            "random_state": 42
        },
        "classify": false,
        "train": true,
        "predict": true
    },
    "label": [
        {
            "filename": "./dataset/test_file_1",
            "label": "malware"
        },
        {
            "filename": "./dataset/test_file_2",
            "label": "benignware"
        }
    ]
}
```

## Configuration Description

### `config`
- Allows inputting only the content of `config`, such as:
```json
{
    "path": {
        "input_file": "./input_file.csv",
        "train_result": "./output/model/score.json",
        "predict_result": "./output/predict/predict_result.json",
        "input_model": "",
        "output_model": "./output/model/SVM.pkl",
        "top_features": "./src/resource/top_features_1.npy",
        "logging_config": "./src/config/logging_config.json",
        "log_file": "./output/log/logging.log"
    },
    "folder": {
        "output": "./output",
        "model": "./output/model",
        "predict": "./output/predict",
        "log": "./output/log"
    },
    "model": {
        "model_name": "SVM",
        "kernel_type": "linear",
        "probability": true,
        "test_size": 0.3,
        "random_state": 42
    },
    "classify": false,
    "train": true,
    "predict": true
}
```

### `config.path`

- `input_file`: Path to the input file, default is `null`, only enabled when `config` is input separately.
- `train_result`: Path to the output file for training results.
- `predict_result`: Path to the output file for prediction results.
- `input_model`: Path to the pre-trained model.
- `output_model`: Path to the output file for the trained model.
- `top_features`: Path to the features file, should point to `top_features_1.npy`.
- `logging_config`: Path to the logging configuration file, should point to `logging_config.json`.
- `log_file`: Path to the output log file.

### `config.folder`

- `output`: Path to the output folder.
- `model`: Path to the model folder.
- `predict`: Path to the prediction results folder.
- `log`: Path to the log folder.

### `config.model`

- `model_name`: Name of the model, should be `SVM`.
- `kernel_type`: Type of SVM kernel function, default is `linear`.
- `probability`: Whether to enable probability estimation, default is `true`.
- `test_size`: Proportion of the test set, default is `0.3`.
- `random_state`: Random seed, default is `42`.

### `config.classify`

- `classify`: Whether to perform classification, currently only `false`.

### `config.train`

- `train`: Whether to perform training.

### `config.predict`

- `predict`: Whether to perform prediction.

### `label`

- `filename`: Path to the labeled file.
- `label`: Label of the file, `malware` or `benignware`.

## Usage

1. Edit the `config_SVM.json` file and modify the configuration as needed.
2. Run `main.py` for model training or prediction.
```sh
python src/code/main.py src/config/config_SVM.py
```

## Notes

- Ensure all paths and folders exist and have appropriate read/write permissions.
- Adjust model parameters as needed to achieve the best results.