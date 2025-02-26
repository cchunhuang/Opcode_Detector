import numpy as np
import joblib as jb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

from Logger import setup_logger 
from utils import Vectorize, Extraction  

class XGBoost:
    '''
    XGBoost classifier for malware detection.
    This class handles data extraction, vectorization, training, and prediction using XGBoost.
    '''
    
    def __init__(self, n_estimators=100, max_depth=6, eta=0.3, test_size=0.3, random_state=42, eval_metric='logloss', top_features_path="./top_features_1.npy"):
        '''
        Initializes the XGBoost classifier with hyperparameters.
        
        Parameters:
        - n_estimators: Number of trees in the XGBoost model
        - max_depth: Maximum depth of each tree
        - eta: Learning rate
        - test_size: Proportion of data used for testing
        - random_state: Seed for reproducibility
        - eval_metric: Evaluation metric used in training
        - top_features_path: Path to the top features used for vectorization
        '''
        self.logger = setup_logger("XGBoost")  # Set up logging for debugging and tracking
        self.logger.info("XGBoost model initialized")
        
        # Initialize the XGBoost classifier with specified parameters
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, eta=eta, 
                                       eval_metric=eval_metric, use_label_encoder=False)
        
        # Store configuration parameters
        self.test_size = test_size
        self.random_state = random_state
        self.top_features_path = top_features_path
    
    def load_data(self, file_list):
        '''
        Loads and processes data from a list of file paths.
        
        Parameters:
        - file_list: List of file paths containing malware or benignware
        
        Returns:
        - vectorized_samples: Feature matrix for classification
        - labels: Corresponding labels (1 for malware, 0 for benignware)
        '''
        self.logger.info(f"Loading data from {file_list}")
        
        # Extract features from each file using the Extraction class
        samples = [Extraction(f) for f in file_list]
        
        # Convert extracted features into a vectorized form
        vectorized_samples = np.vstack([Vectorize(s, self.top_features_path) for s in samples])
        
        # Generate labels based on file path (if the filename contains "malware", label it as 1, else 0)
        labels = [1 if "malware" in f else 0 for f in file_list]  
        
        return vectorized_samples, np.array(labels)
    
    def train(self, file_list, output_model_path=None):
        '''
        Trains the XGBoost model using the provided dataset.
        
        Parameters:
        - file_list: List of file paths to be used for training
        - output_model_path: Path to save the trained model (optional)
        '''
        self.logger.info(f"Training XGBoost model with {file_list}")
        
        # Load and prepare training data
        X, y = self.load_data(file_list)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        # Train the XGBoost classifier
        self.model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        
        # Evaluate model performance using precision, recall, and accuracy
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log evaluation metrics
        self.logger.info(f"Precision: {precision:.2f}")
        self.logger.info(f"Recall: {recall:.2f}")
        self.logger.info(f"Accuracy: {accuracy:.2f}")
        
        # Save the model if an output path is provided
        if output_model_path is not None:
            self.save_model(output_model_path)
    
    def predict(self, file_list):
        '''
        Predicts whether the given files are malware or benignware.
        
        Parameters:
        - file_list: List of file paths to classify
        
        Returns:
        - results: A list of dictionaries with file names and their predicted labels
        '''
        self.logger.info(f"Predicting for files: {file_list}")
        results = []
        
        for filename in file_list:
            self.logger.info(f"Processing {filename}")
            
            # Extract features from the file
            sequence = Extraction(filename)
            
            # Convert features into vectorized format
            X_sample = Vectorize(sequence, self.top_features_path)
            
            # Make a prediction using the trained model
            prediction = self.model.predict(X_sample)
            
            # Convert numerical prediction to human-readable label
            label = "malware" if prediction[0] == 1 else "benignware"
            
            # Store the result
            results.append({"name": filename, "detection": label})
        
        return results
    
    def save_model(self, output_model_path):
        '''
        Saves the trained XGBoost model to a file.
        
        Parameters:
        - output_model_path: Path to save the model
        '''
        self.logger.info(f"Saving model to {output_model_path}")
        jb.dump(self.model, output_model_path)
    
    def load_model(self, input_model_path):
        '''
        Loads a pre-trained XGBoost model from a file.
        
        Parameters:
        - input_model_path: Path to the saved model file
        '''
        self.logger.info(f"Loading model from {input_model_path}")
        self.model = jb.load(input_model_path)
        
if __name__ == "__main__":
    # Define the path to the top features file
    top_features_path = "./src/code/top_features_1.npy"
    
    # List of files to process (both malware and benignware)
    file_list = [
        "OLD/TestingBin/malware/00a0e4105fbecdb5aa33e7cad7edfaecb2e983e0829e0b30eabe384889f107d4", 
        "OLD/TestingBin/malware/00a2bd396600e892da75c686be60d5d13e5a34824afd76c02e8cf7257d2cf5c5", 
        "OLD/TestingBin/benignware/2b0f5e9c8b80c32e42cb4f9924ccd379a9380b88acb6ec6a4bb5ac4d3e952938"
    ]
    
    # Define the path to save or load the trained model
    model_path = './xgboost_model.pkl'
    
    # Create an instance of the XGBoostDetector class
    detector = XGBoost(top_features_path=top_features_path)
    
    # Load a pre-trained model if available
    detector.load_model(model_path)
    
    # Train the model with the provided dataset
    detector.train(file_list)
    
    # Save the trained model to the specified path
    detector.save_model(model_path)
    
    # Run predictions on the provided file list and print results
    print(detector.predict(file_list))
