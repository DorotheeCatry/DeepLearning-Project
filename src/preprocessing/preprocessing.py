import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization, StringLookup, CategoryEncoding
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Preprocess the data for both TensorFlow and scikit-learn models.
    """
    # 1. Create label encoder for target
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # 2. Identify numeric and categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')

    # 3. Create preprocessor for scikit-learn
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

    # 4. Create TensorFlow preprocessing layers
    preprocessing_layers = {}

    # Numeric features
    for col in num_cols:
        normalizer = Normalization()
        normalizer.adapt(X_train[col].values.reshape(-1, 1))
        preprocessing_layers[col] = normalizer

    # Categorical features
    for col in cat_cols:
        lookup = StringLookup(output_mode='int', vocabulary=np.unique(X_train[col]))
        onehot = CategoryEncoding(output_mode='binary', num_tokens=lookup.vocabulary_size())
        preprocessing_layers[col] = (lookup, onehot)

    # 5. Create feature names for processed data
    feature_names = (
        num_cols + 
        [f"{col}_{val}" for col, vals in zip(cat_cols, categorical_transformer.categories) 
         for val in vals[1:]]  # Skip first category due to drop='first'
    )

    return (X_train_processed, X_val_processed, X_test_processed,
            preprocessing_layers, preprocessor, feature_names,
            y_train_enc, y_val_enc, y_test_enc)
