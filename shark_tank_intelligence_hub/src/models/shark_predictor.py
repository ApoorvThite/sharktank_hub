import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import pickle

class SharkPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.shark_columns = []
        
    def prepare_data(self, df, shark_cols):
        self.shark_columns = shark_cols
        
        feature_cols = [col for col in df.columns if col not in shark_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = numeric_cols
        
        X = df[self.feature_columns].fillna(0)
        y = df[shark_cols].fillna(0).astype(int)
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        base_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
        
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X[self.feature_columns].fillna(0))
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_sharks(self, startup_features):
        predictions, probabilities = self.predict(startup_features)
        
        results = []
        for i, shark in enumerate(self.shark_columns):
            results.append({
                'shark': shark,
                'will_invest': bool(predictions[0][i]),
                'probability': probabilities[i][0][1] if len(probabilities[i][0]) > 1 else 0
            })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'shark_columns': self.shark_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.shark_columns = model_data['shark_columns']
        print(f"Model loaded from {filepath}")
