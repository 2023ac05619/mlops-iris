import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR


class DataPipeline:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    def load_iris_dataset(self):
        iris = load_iris()
        self.feature_names = iris.feature_names
        self.target_names = list(iris.target_names)
        return iris.data, iris.target
        
    def save_raw_data(self, X, y):
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = y
        df['target_name'] = [self.target_names[i] for i in y]
        df.to_csv(DATA_DIR / "iris_full.csv", index=False)
        print(f"[INFO] Raw data saved to {DATA_DIR / 'iris_full.csv'}")
        
    def split_and_scale_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def load_and_preprocess(self):
        print("[INFO] Loading Iris dataset...")
        X, y = self.load_iris_dataset()
        
        print("[INFO] Saving raw data...")
        self.save_raw_data(X, y)
        
        print("[INFO] Splitting and scaling data...")
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
        
        print(f"[INFO] Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
