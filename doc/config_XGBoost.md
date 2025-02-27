# config_XGBoost.json

`config_XGBoost.json` is a configuration file for setting up the XGBoost malware detection model. This file includes settings for model training, prediction, and related paths and parameters.

## Configuration Structure

```json
{
    "config": {
        "path": {
            "input_file": null,
            "train_result": "./output/model/score.json",
            "predict_result": "./output/predict/predict_result.json",
            "input_model": "",
            "output_model": "./output/model/XGBoost.pkl",
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
            "model_name": "XGBoost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "use_label_encoder": false,
            "random_state": 42,
            "test_size": 0.3
        },
        "classify": false,
        "train": true,
        "predict": true
    },
    "label": [
        {
            "filename": "./dataset/file_1",
            "label": "malware",
            "tags": "train"
        },
        {
            "filename": "./dataset/file_2",
            "label": "benignware",
            "tags": "test"
        },
        {
            "filename": "./dataset/file_3",
            "label": "malware",
            "tags": "predict"
        }
    ]
}
```

## Configuration Description

### `config`
- Allows inputting only the `config` content, for example:
```json
{
    "path": {
        "input_file": "./input_file.csv",
        "train_result": "./output/model/score.json",
        "predict_result": "./output/predict/predict_result.json",
        "input_model": "",
        "output_model": "./output/model/XGBoost.pkl",
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
        "model_name": "XGBoost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "use_label_encoder": false,
        "random_state": 42,
        "test_size": 0.3
    },
    "classify": false,
    "train": true,
    "predict": true
}
```

### `config.path`

- `input_file`: Path to the input file, default is `null`, enabled only when inputting `config` separately.
- `train_result`: Path to the training result output file.
- `predict_result`: Path to the prediction result output file.
- `input_model`: Path to the pre-trained model.
- `output_model`: Path to the trained model output file.
- `top_features`: Path to the features file, should point to `top_features_1.npy`.
- `logging_config`: Path to the logging configuration file, should point to `logging_config.json`.
- `log_file`: Path to the log output file.

### `config.folder`

- `output`: Path to the output folder.
- `model`: Path to the model folder.
- `predict`: Path to the prediction result folder.
- `log`: Path to the log folder.

### `config.model`

- `model_name`: Model name, should be `XGBoost`.
- `n_estimators`: Number of trees, default is `100`.
- `max_depth`: Maximum depth of the trees, default is `6`.
- `learning_rate`: Learning rate, default is `0.1`.
- `eval_metric`: Evaluation metric, default is `logloss`.
- `use_label_encoder`: Whether to use label encoder, default is `false`.
- `random_state`: Random seed, default is `42`.
- `test_size`: Test set proportion, default is `0.3`.

### `config.classify`

- `classify`: Whether to perform classification, currently only `false`.

### `config.train`

- `train`: Whether to perform training.

### `config.predict`

- `predict`: Whether to perform prediction.

### `label`

- `filename`: Path to the labeled file.
- `label`: Label of the file, `malware` or `benignware`.
- `tags`: Type of the file tag, `train`, `test`, or `predict`

## Usage

1. Edit the `config_XGBoost.json` file and modify the configuration as needed.
2. Run `main.py` for model training or prediction.

```sh
python src/code/main.py src/config/config_XGBoost.py
```

## Notes

- Ensure all paths and folders exist and have appropriate read/write permissions.
- Adjust model parameters as needed to achieve the best results.