import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess(X_train, X_val, X_test, y_train, y_val, y_test):
    

    # LabelEncoding of the target 
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoding = le.fit_transform(y_test)
    y_val_encoding = le.fit_transform(y_val)

    # Categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Numerical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing des colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )

    # Pipeline complète
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.fit_transform(X_test)
    X_val_processed = pipeline.fit_transform(X_val)

    return X_train_processed, X_test_processed, X_val_processed, y_test_encoding, y_train_encoded, y_val_encoding, pipeline, le
