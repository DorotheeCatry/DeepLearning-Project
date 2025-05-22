import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(df, target='Churn', test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the data into training, validation, and test sets with stratification.

    Args:
        df: pandas DataFrame with the data
        target: Name of the target column
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Drop customerID if present (non-informative)
    if 'customerID' in df.columns:
        X = df.drop(columns=['customerID', target])
    else:
        X = df.drop(columns=[target])
    
    y = df[target]

    # Split train+val and test sets (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train into train and validation sets (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
    )

    # Affichage des shapes
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Vérification de la distribution des classes (stratification)
    print("\nClass distribution:")
    print(f"Training set: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Validation set: {y_val.value_counts(normalize=True).to_dict()}")
    print(f"Test set: {y_test.value_counts(normalize=True).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test
