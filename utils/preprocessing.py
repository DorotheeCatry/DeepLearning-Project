import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Preprocess the data for model training and evaluation.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training target
        y_val: Validation target
        y_test: Test target
        
    Returns:
        Processed datasets and preprocessing objects
    """
    # Label encoding for target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    # Identify categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove customerID from features if present
    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed validation data shape: {X_val_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, X_val_processed, y_train_encoded, y_val_encoded, y_test_encoded, preprocessor, le