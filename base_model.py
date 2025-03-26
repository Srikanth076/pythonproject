# base_model.py
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def print_results(self, X_test, y_test):
        pass