import os
import csv
import json
import numpy as np
import joblib as jb
from box import Box
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

from Logger import setup_logger
from utils import Vectorize, Extraction

class XGBoost():
    '''
    XGBoost classifier for malware detection.
    This class handles data extraction, vectorization, training, and prediction using XGBoost.
    '''
    
    def __init__(self, config_path="./config.json"):
        '''
        Initializes the XGBoost classifier.
        '''
        # Read config.json
        with open(config_path) as f:
            self.config = Box(json.load(f))
            
        if hasattr(self.config, "config"):
            self.file_list = {f["filename"]: f["label"] for f in self.config.label}
            self.config = self.config.config
        else:
            self.file_list = {}
            with open(self.config.path.input_file, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    self.file_list[row['filename']] = row['label']
        
        # Create necessary directories
        for folder_name, folder_path in self.config.folder.items():
            os.makedirs(folder_path, exist_ok=True)
        
        # Set up logging system
        self.logger = setup_logger(logger_name="XGBoost", logging_config_path=self.config.path.logging_config, output_log_path=self.config.path.log_file)
        self.logger.info("XGBoost model initialized")
        
        # Initialize XGBoost model
        self.xgboost = xgb.XGBClassifier(n_estimators=self.config.model.n_estimators, max_depth=self.config.model.max_depth, eta=self.config.model.learning_rate,
                                       eval_metric=self.config.model.eval_metric, use_label_encoder=self.config.model.use_label_encoder)
        
    def extractFeature(self, filename=None):
        '''
        Extracts features from the malware dataset.
        
        Parameters:
        filename (str): Path to the file to extract features from (default: None, using file_list from config).

        Returns:
        list: A list of extracted features from the dataset(s).
        '''
        if filename is not None:
            self.logger.info(f"Extracting features")
            return Extraction(filename)
        
        self.logger.info(f"Extracting features from config file_list")
        return [Extraction(f) for f in self.file_list.keys()]
    
    def vectorize(self, sequence):
        '''
        Vectorizes the extracted features into a numpy array.
        
        Parameters:
        sequence (list): List of extracted features.

        Returns:
        numpy array: Vectorized form of the extracted features.
        '''
        self.logger.info("Vectorizing extracted features")
        return Vectorize(sequence, self.config.path.top_features)
        
    def load_data(self, file_list=None):
        '''
        Loads and processes data from a list of file paths.
        
        Parameters:
        file_list (list): List of file paths to be used for training or prediction.(default: None, using file_list from config)

        Returns:
        tuple: A tuple containing:
            - numpy array of vectorized features.
            - numpy array of corresponding labels (0 for malware, 1 for benignware).
        '''
        if file_list is None:
            file_list = self.file_list.keys()
            self.logger.info(f"Loading data from config file_list")
        else:
            self.logger.info(f"Loading data from parameter")
        
        samples = self.extractFeature()
        vectorized_samples = np.vstack([self.vectorize(s) for s in samples])
        labels = [1 if label == "benignware" else 0 for label in self.file_list.values()]
        
        return vectorized_samples, np.array(labels)
    
    def model(self, training=True):
        '''
        Trains or loads the XGBoost model.
        
        Parameters:
        training (bool): Whether to train the model (default: True) (False: Load or save model)

        Returns:
        None
        '''
        if self.config.path.input_model:
            self.logger.info(f"Loading model from {self.config.path.input_model}")
            self.load_model(self.config.path.input_model)
        
        if training:
            self.logger.info(f"Training XGBoost model")
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.model.test_size, random_state=self.config.model.random_state)
            
            self.xgboost.fit(X_train, y_train)
            
            y_pred = self.xgboost.predict(X_test)
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Precision: {precision:.2f}")
            self.logger.info(f"Recall: {recall:.2f}")
            self.logger.info(f"Accuracy: {accuracy:.2f}")
            
        if self.config.path.output_model:
            self.save_model(self.config.path.output_model)
    
    def predict(self):
        '''
        Predicts whether the given files are malware or benignware.
        
        Returns:
        list: A list of dictionaries containing the file name and detection result.
        '''
        self.logger.info(f"Predicting")
        results = []
        
        for filename in self.file_list.keys():
            self.logger.info(f"Processing {filename}")
            sequence = self.extractFeature(filename)
            X_sample = self.vectorize(sequence)
            prediction = self.xgboost.predict(X_sample)
            label = "benignware" if prediction[0] == 1 else "malware"
            results.append({"name": filename, "detection": label})
            
        if self.config.path.output_predict_file:
            self.logger.info(f"Saving results to {self.config.path.output_predict_file}")
            with open(self.config.path.output_predict_file, mode='w') as file:
                json.dump(results, file)
        
        return results
    
    def save_model(self, output_model_path):
        '''
        Saves the trained XGBoost model to a file.
        '''
        self.logger.info(f"Saving model to {output_model_path}")
        jb.dump(self.xgboost, output_model_path)
    
    def load_model(self, input_model_path):
        '''
        Loads a pre-trained XGBoost model from a file.
        '''
        self.logger.info(f"Loading model from {input_model_path}")
        self.xgboost = jb.load(input_model_path)
        
if __name__ == "__main__":
    config_path = "./src/config/config_XGBoost.json"
    model = XGBoost(config_path)
    model.model(training=True)
    model.predict()
