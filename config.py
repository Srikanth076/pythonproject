# config.py

class Config:
    # Data file path
    DATA_FILE = 'email_data.csv'
    
    # Random state for reproducibility
    RANDOM_STATE = 42
    
    # TF-IDF parameters
    MAX_FEATURES = 5000
    MIN_DF = 2  # Minimum document frequency
    MAX_DF = 0.95  # Maximum document frequency
    
    # Model parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = None
    MIN_SAMPLES_SPLIT = 2
    MIN_SAMPLES_LEAF = 1
    
    # Multi-label classification parameters
    MIN_CLASS_INSTANCES = 5  # Minimum instances required for a class
    LABEL_TYPES = ['Type2', 'Type3', 'Type4']  # Types of labels to predict
    
    # Data preprocessing
    MIN_CLASS_INSTANCES = 5
    TEST_SIZE = 0.2  # Train-test split ratio