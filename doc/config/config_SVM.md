# config_SVM.json

`config_SVM.json` 是用於配置 SVM (Support Vector Machine) 惡意軟體檢測模型的設定檔。此檔案包含了模型訓練、預測以及相關路徑和參數的設定。

## 配置結構

```json
{
    "config": {
        "path": {
            "input_file": null,
            "output_file": "./output/predict/predict_result.json",
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
            "filename": "./dataset/TestingBin/malware/00a0e4105fbecdb5aa33e7cad7edfaecb2e983e0829e0b30eabe384889f107d4",
            "label": "malware"
        },
        {
            "filename": "./dataset/TestingBin/malware/00a2bd396600e892da75c686be60d5d13e5a34824afd76c02e8cf7257d2cf5c5",
            "label": "malware"
        },
        {
            "filename": "./dataset/TestingBin/benignware/2b0f5e9c8b80c32e42cb4f9924ccd379a9380b88acb6ec6a4bb5ac4d3e952938",
            "label": "benignware"
        }
    ]
}
```

## 配置說明

### `config`
- SVM.py 允許只輸入 config 的內容，如：
```json
{
    "path": {
        "input_file": "./input_file.csv",
        "output_file": "./output/predict/predict_result.json",
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

- `input_file`: 輸入文件的路徑，預設為 `null`，只有單獨輸入 `config` 時才會啟用。
- `output_file`: 預測結果的輸出文件路徑。
- `input_model`: 預訓練模型的輸入路徑。
- `output_model`: 訓練後模型的輸出路徑。
- `top_features`: 特徵文件的路徑，需指向 top_features_1.npy。
- `logging_config`: 日誌配置文件的路徑，需指向 logging_config.json。
- `log_file`: 日誌文件的輸出路徑。

### `config.folder`

- `output`: 輸出文件夾的路徑。
- `model`: 模型文件夾的路徑。
- `predict`: 預測結果文件夾的路徑。
- `log`: 日誌文件夾的路徑。

### `config.model`

- `model_name`: 模型名稱，需為 `SVM`。
- `kernel_type`: SVM 核函數類型，預設為 `linear`。
- `probability`: 是否啟用概率估計，預設為 `true`。
- `test_size`: 測試集所佔比例，預設為 `0.3`。
- `random_state`: 隨機種子，預設為 `42`。

### `config.classify`

- `classify`: 是否進行分類，目前只有 `false`。

### `config.train`

- `train`: 是否進行訓練。

### `config.predict`

- `predict`: 是否進行預測。

### `label`

- `filename`: 標註文件的路徑。
- `label`: 文件的標籤，`malware` 或 `benignware`。

## 使用方法

1. 編輯 config_SVM.json 文件，根據需要修改配置。
2. 執行 SVM.py 進行模型訓練或預測。

```sh
python src/code/SVM.py
```

3. 或執行 main.py
```sh
python src/code/main.py src/config/config_SVM.py
```

## 注意事項

- 確保所有路徑和文件夾存在，並且具有適當的讀寫權限。
- 根據需要調整模型參數以獲得最佳效果。
