# data_preprocessor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def load_data(self):
        """Load and clean data"""
        self.data = pd.read_csv(self.filepath)
        # Assuming columns: Type1, Type2, Type3, Type4, Text
        # Type1 is ignored as per requirement
        return self.data.drop('Type1', axis=1)
    
    def remove_rare_classes(self, df, column):
        """Remove classes with few instances"""
        value_counts = df[column].value_counts()
        valid_classes = value_counts[value_counts >= Config.MIN_CLASS_INSTANCES].index
        return df[df[column].isin(valid_classes)]
    
    def preprocess(self):
        """Prepare training and testing data"""
        df = self.load_data()
        
        # Clean data for each type
        for col in ['Type2', 'Type3', 'Type4']:
            df = self.remove_rare_classes(df, col)
        
        # Split features and labels
        X = df['Text']
        y = df[['Type2', 'Type3', 'Type4']]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=Config.RANDOM_STATE
        )
        
        return DataContainer(X_train, y_train, X_test, y_test)

class DataContainer:
    """Encapsulate input data consistently"""
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test