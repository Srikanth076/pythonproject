# chained_classifier.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from base_model import BaseClassifier
from config import Config
import numpy as np

class ChainedClassifier(BaseClassifier):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=Config.MAX_FEATURES)
        self.models = {
            'Type2': RandomForestClassifier(random_state=Config.RANDOM_STATE),
            'Type2_3': RandomForestClassifier(random_state=Config.RANDOM_STATE),
            'Type2_3_4': RandomForestClassifier(random_state=Config.RANDOM_STATE)
        }
    
    def _prepare_combined_labels(self, y, levels):
        """Combine labels for chaining"""
        if not isinstance(y, dict) or not all(key in y for key in ['Type2', 'Type3', 'Type4']):
            return None

        if levels == ['Type2']:
            return y['Type2']
        elif levels == ['Type2', 'Type3']:
            return y['Type2'].astype(str) + '_' + y['Type3'].astype(str)
        else:
            return y['Type2'].astype(str) + '_' + y['Type3'].astype(str) + '_' + y['Type4'].astype(str)
    
    def train(self, X, y):
        """Train chained models"""
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Train each level
        self.models['Type2'].fit(X_tfidf, y['Type2'])
        self.models['Type2_3'].fit(X_tfidf, self._prepare_combined_labels(y, ['Type2', 'Type3']))
        self.models['Type2_3_4'].fit(X_tfidf, self._prepare_combined_labels(y, ['Type2', 'Type3', 'Type4']))
    
    def predict(self, X) -> None:
        """Predict using chained models"""
        X_tfidf = self.vectorizer.transform(X)
        pred_2 = self.models['Type2'].predict(X_tfidf)
        pred_2_3 = self.models['Type2_3'].predict(X_tfidf)
        pred_2_3_4 = self.models['Type2_3_4'].predict(X_tfidf)
        # Return tuple of predictions or None if empty
        # Return None if any prediction is empty or contains None values
        if len(pred_2) == 0 or len(pred_2_3) == 0 or len(pred_2_3_4) == 0:
            return None
        predictions = (pred_2, pred_2_3, pred_2_3_4)
        return predictions
    
    def print_results(self, X_test, y_test):
        """Evaluate and print results"""
        predictions = self.predict(X_test)
        if predictions is None:
            pred_2, pred_2_3, pred_2_3_4 = [], [], []
        else:
            pred_2, pred_2_3, pred_2_3_4 = predictions if predictions is not None else ([], [], [])
        
        # Split combined predictions
        actual_2 = y_test['Type2']
        actual_2_3 = self._prepare_combined_labels(y_test, ['Type2', 'Type3'])
        actual_2_3_4 = self._prepare_combined_labels(y_test, ['Type2', 'Type3', 'Type4'])
        
        print("Chained Classifier Results:")
        print(f"Type2 Accuracy: {accuracy_score(actual_2, pred_2):.2f}")
        print(f"Type2+3 Accuracy: {accuracy_score(actual_2_3, pred_2_3):.2f}")
        print(f"Type2+3+4 Accuracy: {accuracy_score(actual_2_3_4, pred_2_3_4):.2f}")