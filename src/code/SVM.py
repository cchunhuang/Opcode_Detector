import numpy as np
import joblib as jb
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

from Logger import setup_logger
from Vectorize import Vectorize
from Extraction import Extraction

class SVM:
    '''
    Support Vector Machine (SVM) classifier for malware detection.
    This class handles data extraction, vectorization, training, and prediction using SVM.
    '''
    
    def __init__(self, kernel_type='linear', probability=True, test_size=0.3, random_state=42, top_features_path="./top_features_1.npy"):
        '''
        Initializes the SVM classifier.

        Parameters:
        kernel_type (str): Kernel type for the SVM model (default: 'linear').
        probability (bool): Whether to enable probability estimates (default: True).
        test_size (float): Proportion of the dataset to include in the test split (default: 0.3).
        random_state (int): Random seed for reproducibility (default: 42).
        top_features_path (str): Path to the top features file (default: "./top_features_1.npy").
        '''
        self.logger = setup_logger("SVM")
        self.logger.info("SVM model initialized")
        
        # Initialize the SVM model
        self.model = svm.SVC(kernel=kernel_type, probability=probability)
        
        # Store parameters
        self.test_size = test_size
        self.random_state = random_state
        self.top_features_path = top_features_path
    
    def load_data(self, file_list):
        '''
        Loads and processes data from a list of file paths.

        Parameters:
        file_list (list): List of file paths to be used for training or prediction.

        Returns:
        tuple: A tuple containing:
            - numpy array of vectorized features.
            - numpy array of corresponding labels (1 for malware, 0 for benignware).
        '''
        self.logger.info(f"Loading data from {file_list}")
        
        # Extract sequences from the given files
        samples = [Extraction(f) for f in file_list]
        
        # Convert sequences into vectorized form using predefined features
        vectorized_samples = np.vstack([Vectorize(s, self.top_features_path) for s in samples])
        
        # Generate labels based on filename (assumption: filenames contain malware/benignware indicators)
        labels = [1 if "malware" in f else 0 for f in file_list]  

        return vectorized_samples, np.array(labels)
    
    def train(self, file_list, output_model_path=None):
        '''
        Trains the SVM model using the provided dataset.

        Parameters:
        file_list (list): List of file paths to be used for training.

        Returns:
        None
        '''
        self.logger.info(f"Training SVM model with {file_list}")
        
        # Load data and split into training and testing sets
        X, y = self.load_data(file_list)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        # Train the SVM model
        self.model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        
        # Compute evaluation metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log evaluation results
        self.logger.info(f"Precision: {precision:.2f}")
        self.logger.info(f"Recall: {recall:.2f}")
        self.logger.info(f"Accuracy: {accuracy:.2f}")
        
        # Save model if an output path is specified
        if output_model_path is not None:
            self.save_model()
    
    def predict(self, file_list):
        '''
        Predicts whether the given files are malware or benignware.

        Parameters:
        file_list (list): List of file paths for prediction.

        Returns:
        list: A list of dictionaries containing the file name and detection result.
        '''
        self.logger.info(f"Predicting for files: {file_list}")
        results = []
        
        for filename in file_list:
            self.logger.info(f"Processing {filename}")
            
            # Extract features from the file
            sequence = Extraction(filename)
            
            # Convert the sequence into a vectorized form
            X_sample = Vectorize(sequence, self.top_features_path)
            
            # Predict using the trained model
            prediction = self.model.predict(X_sample)
            
            # Convert prediction result into a human-readable label
            label = "malware" if prediction[0] == 1 else "benignware"
            
            results.append({"name": filename, "detection": label})
        
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
        jb.dump(self.model, output_model_path)
    
    def load_model(self, input_model_path):
        '''
        Loads a pre-trained SVM model from a file.

        Returns:
        None
        '''
        self.logger.info(f"Loading model from {input_model_path}")
        self.model = jb.load(input_model_path)
        
if __name__ == "__main__":
    # Define path to the feature file
    top_features_path = "./src/code/top_features_1.npy"
    
    # List of sample files for training and testing
    file_list = [
        "OLD/TestingBin/malware/00a0e4105fbecdb5aa33e7cad7edfaecb2e983e0829e0b30eabe384889f107d4", 
        "OLD/TestingBin/malware/00a2bd396600e892da75c686be60d5d13e5a34824afd76c02e8cf7257d2cf5c5", 
        "OLD/TestingBin/benignware/2b0f5e9c8b80c32e42cb4f9924ccd379a9380b88acb6ec6a4bb5ac4d3e952938"
    ]
    
    model_path = './SVM_model.pkl'
    
    # Create an SVM detector instance
    detector = SVM(top_features_path=top_features_path)
    # detector.load_model(model_path)
    
    # Train the model using the sample dataset
    detector.train(file_list)
    detector.save_model(model_path)
    
    # Test the trained model with prediction
    test_file = file_list
    print(detector.predict(test_file))
