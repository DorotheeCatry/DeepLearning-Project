import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    """
    Preprocess the data for the churn prediction model.
    
    Args:
        df: pandas DataFrame with the raw data
        
    Returns:
        Preprocessed features and encoded target
    """
    # Separate features and target
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']
    
    # Identify numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])
    
    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X_processed, y_encoded, preprocessor, le