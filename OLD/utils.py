import os
import json
import argparse
from types import SimpleNamespace

def parameter_parser():
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("config", type=str, help="Path to the JSON file")
    return parser.parse_args()

def load_json(path):
    '''
    param. path: str, path to the config file
    return. config: dict, the content of the config file
    '''
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
        print("JSON Data Loaded Successfully:")
        return data
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{path}' is not a valid JSON file.")

def write_output(input_path,output_path,result,labels):
    '''
    param.
        input_path: path to the input binary file
        output_path: path to the csv file
        result: a list(float), probability of each class, e.g. [0.92, 0.08]
        labels: a list(str), each class, e.g. ['Benignware', 'Malware']
    description.
        write a csv file for saving the result of prediction
            e.g.
                FILENAME, Benign, Malware
                34eff01a, 0.98, 0.02
                f01a34ef, 0.17, 0.83
                ff001aff, -1
            or e.g.
                FILENAME, Benign, Mirai, Unknown, Android
                34eff01a, 0.01, 0.02, 0.95, 0.02
                f01a34ef, 0.73, 0, 0.22, 0.05
                ff001aff, -1
    '''

    if '.csv' not in output_path:
        output_path += '.csv'

    # init the columns of this table if it is a new csv file
    if not os.path.exists(output_path):
        with open(output_path,'w') as f:
            line = 'Filename, '
            line += ', '.join(labels)
            line += '\n'
            f.write(line)

    # write the result 
    with open(output_path,'a+') as f:
        line = os.path.basename(input_path) + ', '
        line += ','.join(list(str(i) for i in result))
        line += '\n'
        f.write(line)