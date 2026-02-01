import json
import argparse

from MalwareDetector import MalwareDetector

def main(args):
    with open(args.config_path) as f:
        config = json.load(f)
        
    model = MalwareDetector(args.config_path)
    
    action = config.get("action", "train")
    
    if action == "train":
        model.get_model(action="train")
        model.get_model(action="predict")
        model.get_prediction()
    elif action == "predict":
        model.get_model(action="predict")
        model.get_prediction()
    else:
        raise ValueError(f"Invalid action: {action}. Must be 'train' or 'predict'")
    
def parameter_parser():
    parser = argparse.ArgumentParser(description="Malware Detection")
    parser.add_argument("config_path", nargs='?', default="./config.json", help="Path to the configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parameter_parser()
    main(args)