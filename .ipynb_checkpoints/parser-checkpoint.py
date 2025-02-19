import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Opcode n-gram Detector.")

    parser.add_argument('--input-path',
                        nargs='?',
                        default='./TestingBin/malware/00a0e4105fbecdb5aa33e7cad7edfaecb2e983e0829e0b30eabe384889f107d4',
                        help='input binary file.')

    parser.add_argument('--model',
                        nargs='?',
                        default='xgboost',
                        help='Select the model(xgboost , SVM).')
    
    parser.add_argument('--MDorFC',
                        nargs='?',
                        default='MD',
                        help='if it is expected to classify the family of malware, type \'--MDorFC FC\', then the program will import the FC model.')

    return parser.parse_args()