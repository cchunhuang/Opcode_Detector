import sys
import json
import argparse

# from SVM import SVM
# from XGBoost import XGBoost
from MalwareDetector import MalwareDetector

def main(args):
    with open(args.config_path) as f:
        config = json.load(f)
    
    if "config" in config:
        config = config["config"]
        
    model = MalwareDetector(args.config_path)
        
    # if config["model"]["model_name"] == "SVM":
    #     model = SVM(args.config_path)
    # elif config["model"]["model_name"] == "XGBoost":
    #     model = XGBoost(args.config_path)
    # else:
    #     raise ValueError(f"Invalid model name: {config['model']['model_name']}")
    
    if config["train"]:
        model.model(training=True)
    if config["predict"]:
        model.model(training=False)
        model.predict()
    if not config["train"] and not config["predict"]:
        raise ValueError("Invalid operation: Must specify either 'train' or 'predict'")
    
def parameter_parser():
    parser = argparse.ArgumentParser(description="Malware Detection")
    parser.add_argument("config_path", nargs='?', default="./config.json", help="Path to the configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parameter_parser()
    main(args)