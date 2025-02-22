import numpy as np
import joblib as jb
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

from Extraction import Extraction
from Vectorize import vectorize

class SVM:
    def __init__(self, kernel_type='linear', probability=True, test_size=0.3, random_state=42, ngram_range=(2, 4), top_features_path="./top_features_1.npy"):
        self.model = svm.SVC(kernel=kernel_type, probability=probability)
        self.test_size = test_size
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.top_features_path = top_features_path
    
    def load_data(self, file_list):
        samples = [Extraction(f) for f in file_list]
        vectorized_samples = np.vstack([vectorize(s, self.ngram_range, self.top_features_path) for s in samples])
        labels = [1 if "malware" in f else 0 for f in file_list]  # 假設文件名稱包含類別標籤
        return vectorized_samples, np.array(labels)
    
    def train(self, file_list):
        X, y = self.load_data(file_list)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
    
    def predict(self, filename):
        sequence = Extraction(filename)
        X_sample = vectorize(sequence, self.ngram_range, self.top_features_path)
        return self.model.predict(X_sample)
    
    def save_model(self):
        jb.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        self.model = jb.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
        
if __name__ == "__main__":
    file_list = ["sample1.exe", "sample2.exe"]  # 替換為實際文件列表
    detector = SVM()
    detector.train(file_list)
    test_file = "test_sample.exe"
    print(f"Prediction for {test_file}:", detector.predict(test_file))