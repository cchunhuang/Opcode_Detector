import os
import csv
import json
import numpy as np
import joblib as jb
from box import Box
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

from Logger import setup_logger
from utils import Vectorize, Extraction

class SVM():
    '''
    Support Vector Machine (SVM) classifier for malware detection.
    This class handles data extraction, vectorization, training, and prediction using SVM.
    '''
    
    def __init__(self, config_path="./config.json"):
        '''
        Initializes the SVM classifier.
        '''
        # Read config.json
        with open(config_path) as f:
            self.config = Box(json.load(f))
            
        if hasattr(self.config, "config"):
            # Settings for Platform
            self.file_list = {}
            for f in self.config.label:
                self.file_list[f["filename"]] = f["label"]
            self.config = self.config.config
        else:
            # Settings for Local
            self.file_list = {}
            with open(self.config.path.input_file, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    self.file_list[row['filename']] = row['label']
                    
        # Initialize folders
        for folder_name, folder_path in self.config.folder.items():
            os.makedirs(folder_path, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(logger_name="SVM", logging_config_path=self.config.path.logging_config, output_log_path=self.config.path.log_file)
        self.logger.info("SVM model initialized")
        
        # Initialize the SVM model
        self.svm = svm.SVC(kernel=self.config.model.kernel_type, probability=self.config.model.probability)
        
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
            - numpy array of corresponding labels (1 for malware, 0 for benignware).
        '''
        if file_list is None:
            file_list = self.file_list.keys()
            self.logger.info(f"Loading data from config file_list")
        else:
            self.logger.info(f"Loading data from parameter")
        
        # Extract sequences from the given files
        samples = self.extractFeature()
        
        # Convert sequences into vectorized form using predefined features
        vectorized_samples = np.vstack([self.vectorize(s) for s in samples])
        
        # Generate labels based on filename (assumption: filenames contain malware/benignware indicators)
        labels = [1 if label=="benignware" else 0  for label in self.file_list.values()]

        return vectorized_samples, np.array(labels)
    
    def model(self, training=True):
        '''
        Trains the SVM model using the provided dataset.

        Parameters:
        training (bool): Whether to train the model (default: True) (False: Load or save model)

        Returns:
        None
        '''
        # Load pre-trained model if available
        if self.config.path.input_model != "" and self.config.path.input_model is not None:
            self.logger.info(f"Loading model from {self.config.path.input_model}")
            self.load_model(self.config.path.input_model)
        
        if training:
            file_list = self.file_list.keys()
            self.logger.info(f"Training SVM model")
            
            # Load data and split into training and testing sets
            X, y = self.load_data(file_list)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.model.test_size, random_state=self.config.model.random_state)
        
            # Train the SVM model
            self.svm.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred = self.svm.predict(X_test)
            
            # Compute evaluation metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log evaluation results
            self.logger.info(f"Precision: {precision:.2f}")
            self.logger.info(f"Recall: {recall:.2f}")
            self.logger.info(f"Accuracy: {accuracy:.2f}")
            
        # Save model if an output path is specified
        if self.config.path.output_model != "" and self.config.path.output_model is not None:
            self.save_model(self.config.path.output_model)
    
    def predict(self):
        '''
        Predicts whether the given files are malware or benignware.

        Parameters:
        file_list (list): List of file paths for prediction.

        Returns:
        list: A list of dictionaries containing the file name and detection result.
        '''
        file_list = self.file_list.keys()
        self.logger.info(f"Predicting")
        results = []
        
        for filename in file_list:
            self.logger.info(f"Processing {filename}")
            
            # Extract features from the file
            sequence = self.extractFeature(filename)
            
            # Convert the sequence into a vectorized form
            X_sample = self.vectorize(sequence)
            
            # Predict using the trained model
            prediction = self.svm.predict(X_sample)
            
            # Convert prediction result into a human-readable label
            label = "benignware" if prediction[0] == 1 else "malware"
            
            results.append({"name": filename, "detection": label})
            
        if self.config.path.output_file != "" and self.config.path.output_file is not None:
            self.logger.info(f"Saving results to {self.config.path.output_file}")
            with open(self.config.path.output_file, mode='w') as file:
                json.dump(results, file)
        
        return results
    
    def save_model(self, output_model_path):
        '''
        Saves the trained SVM model to a file.
        
        Parameters:
        output_model_path (str): Path to save the trained

        Returns:
        None
        '''
        self.logger.info(f"Saving model to {output_model_path}")
        jb.dump(self.svm, output_model_path)
    
    def load_model(self, input_model_path):
        '''
        Loads a pre-trained SVM model from a file.

        Returns:
        None
        '''
        self.logger.info(f"Loading model from {input_model_path}")
        self.svm = jb.load(input_model_path)
        
        
if __name__ == "__main__":
    config_path = "./src/config/config_SVM.json"
    model = SVM(config_path)
    model.model(training=True)
    model.predict()