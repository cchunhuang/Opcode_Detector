import os
import csv
import json
import random

# -----parameters-----

train_num = 6000
predict_num = 100

train = True
predict = True

train_ratio = 0.8

label_path = './dataset/dataset202312/dataset.csv'
data_folder = './dataset/dataset202312/data/'

input_config_paths = ['./src/config/config_SVM.json', './src/config/config_XGBoost.json']
output_config_paths = ['./output/config/config_SVM.json', './output/config/config_XGBoost.json']

# -----label-----

with open(label_path, 'r', newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    selected_files = [row for row in rows]
    
for data in selected_files:
    data["filename"] = os.path.join(data_folder, data["filename"])
    
random.seed(42)

# -----train-----

random.shuffle(selected_files)
train_files = selected_files[:train_num]

i = 0
train_num = int(train_num * train_ratio)
for data in train_files:
    if i < train_num:
        data["tags"] = "train"
        i += 1
    else:
        data["tags"] = "test"
        
# -----predict-----

random.shuffle(selected_files)
predict_label = selected_files[:predict_num]

for data in predict_label:
    data["tags"] = "predict"

# -----output-----

for input_config_path, output_config_path in zip(input_config_paths, output_config_paths):

    with open(input_config_path, 'r') as f:
        config = json.load(f)
        
    config = config['config']

    config['train'] = train
    config['predict'] = predict
        
    combined_data = {
        "config": config,
        "label": train_files + predict_label
    }
    
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, 'w') as f:
        json.dump(combined_data, f, indent=4)