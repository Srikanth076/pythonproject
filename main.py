# main.py
from data_preprocessor import DataPreprocessor
from chained_classifier import ChainedClassifier
from hierarchical_classifier import HierarchicalClassifier
from config import Config

def main():
    # Preprocessing
    preprocessor = DataPreprocessor(Config.DATA_FILE)
    data = preprocessor.preprocess()
    
    # Chained Classifier
    chained = ChainedClassifier()
    chained.train(data.X_train, data.y_train)
    chained.print_results(data.X_test, data.y_test)
    
    print("\n" + "="*50 + "\n")
    
    # Hierarchical Classifier
    hierarchical = HierarchicalClassifier()
    hierarchical.train(data.X_train, data.y_train)
    hierarchical.print_results(data.X_test, data.y_test)

if __name__ == "__main__":
    main()