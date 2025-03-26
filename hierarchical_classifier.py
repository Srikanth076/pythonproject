# hierarchical_classifier.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from base_model import BaseClassifier
from config import Config

class HierarchicalClassifier(BaseClassifier):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=Config.MAX_FEATURES)
        self.type2_model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
        self.type3_models = {}  # One model per Type2 class
        self.type4_models = {}  # One model per Type2+Type3 combination
    
    def train(self, X, y):
        """Train hierarchical models"""
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Level 1: Type2
        self.type2_model.fit(X_tfidf, y['Type2'])
        
        # Level 2: Type3 models per Type2 class
        type2_preds = self.type2_model.predict(X_tfidf)
        for type2_class in set(type2_preds):
            mask = type2_preds == type2_class
            X_subset = X_tfidf[mask]
            y_subset = y['Type3'][mask]
            if len(y_subset) > 0:
                model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
                model.fit(X_subset, y_subset)
                self.type3_models[type2_class] = model
        
        # Level 3: Type4 models per Type2+Type3 combination
        for type2_class, type3_model in self.type3_models.items():
            mask = type2_preds == type2_class
            type3_preds = type3_model.predict(X_tfidf[mask])
            for type3_class in set(type3_preds):
                sub_mask = type3_preds == type3_class
                X_sub_subset = X_tfidf[mask][sub_mask]
                y_sub_subset = y['Type4'][mask][sub_mask]
                if len(y_sub_subset) > 0:
                    model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
                    model.fit(X_sub_subset, y_sub_subset)
                    self.type4_models[f"{type2_class}_{type3_class}"] = model
    
    def predict(self, X) -> None:
        """Predict using hierarchical models"""
        X_tfidf = self.vectorizer.transform(X)
        
        # Level 1 prediction
        pred_2 = self.type2_model.predict(X_tfidf)
        
        # Level 2 prediction
        pred_3 = []
        for i, type2_pred in enumerate(pred_2):
            if type2_pred in self.type3_models:
                pred = self.type3_models[type2_pred].predict(X_tfidf[i:i+1])
                pred_3.append(pred[0])
            else:
                pred_3.append(None)
        
        # Level 3 prediction
        pred_4 = []
        for i, (type2_pred, type3_pred) in enumerate(zip(pred_2, pred_3)):
            key = f"{type2_pred}_{type3_pred}"
            if key in self.type4_models:
                pred = self.type4_models[key].predict(X_tfidf[i:i+1])
                pred_4.append(pred[0])
            else:
                pred_4.append(None)
        
        return pred_2, pred_3, pred_4
    
    def print_results(self, X_test, y_test):
        """Evaluate and print results"""
        try:
            # Handle case where predict returns None
            result = self.predict(X_test)
            if result is None:
                pred_2, pred_3, pred_4 = [], [], []
            else:
                if isinstance(result, tuple) and len(result) == 3:
                    pred_2, pred_3, pred_4 = list(result[0]), list(result[1]), list(result[2])
                else:
                    pred_2, pred_3, pred_4 = [], [], []
        except TypeError:
            # Handle case where predict returns None
            pred_2, pred_3, pred_4 = [], [], []
        
        print("Hierarchical Classifier Results:")
        print(f"Type2 Accuracy: {accuracy_score(y_test['Type2'], pred_2):.2f}")
        
        valid_3 = [i for i, p in enumerate(pred_3) if p is not None]
        if valid_3:
            print(f"Type3 Accuracy: {accuracy_score(y_test['Type3'].iloc[valid_3], [pred_3[i] for i in valid_3]):.2f}")
        
        valid_4 = [i for i, p in enumerate(pred_4) if p is not None]
        if valid_4:
            print(f"Type4 Accuracy: {accuracy_score(y_test['Type4'].iloc[valid_4], [pred_4[i] for i in valid_4]):.2f}")