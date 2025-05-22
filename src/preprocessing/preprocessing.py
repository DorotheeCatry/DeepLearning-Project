import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Preprocess the data for both TensorFlow and scikit-learn models.
    """
    # Create label encoder for target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # Identify numeric and categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')

    # Create preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return (X_train_processed, X_val_processed, X_test_processed,
            y_train_enc, y_val_enc, y_test_enc)