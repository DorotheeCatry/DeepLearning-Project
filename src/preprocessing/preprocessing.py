import pandas as pd
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization, StringLookup, CategoryEncoding

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Preprocess the data for TensorFlow model training and evaluation.
    
    Args:
        X_train: pd.DataFrame, training features
        X_val: pd.DataFrame, validation features
        X_test: pd.DataFrame, test features
        y_train: pd.Series or array-like, training target
        y_val: pd.Series or array-like, validation target
        y_test: pd.Series or array-like, test target
        
    Returns:
        X_train_dict, X_val_dict, X_test_dict: dicts with preprocessed feature arrays (raw data, ready for model)
        y_train, y_val, y_test: encoded targets as numpy arrays
        preprocessing_layers: dict of preprocessing layers keyed by feature name (to integrate in the model)
    """
    # 0. Convertir la cible en binaire
    y_train = y_train.map({'yes': 1, 'no': 0})
    y_val = y_val.map({'yes': 1, 'no': 0})
    y_test = y_test.map({'yes': 1, 'no': 0})
    
    # 1. Encoder les cibles
    y_train_enc = np.array(y_train)
    y_val_enc = np.array(y_val)
    y_test_enc = np.array(y_test)

    # 2. Identifier les colonnes numériques et catégorielles
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Retirer customerID si présent
    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')

    # 3. Construire les preprocessing layers (adaptés uniquement sur train)

    preprocessing_layers = {}

    # Numeriques : Normalization
    for col in num_cols:
        normalizer = Normalization()
        normalizer.adapt(X_train[col].values.reshape(-1, 1))
        preprocessing_layers[col] = normalizer

    # Categoriels : StringLookup + OneHot encoding
    for col in cat_cols:
        lookup = StringLookup(output_mode='int', vocabulary=np.unique(X_train[col]))
        onehot = CategoryEncoding(output_mode='binary', num_tokens=lookup.vocabulary_size())
        preprocessing_layers[col] = (lookup, onehot)

    # 4. Convertir les DataFrames en dict {feature_name: array}
    def df_to_dict(df):
        data_dict = {}
        for col in num_cols:
            data_dict[col] = df[col].values.astype('float32')  # Numeric as float32
        for col in cat_cols:
            data_dict[col] = df[col].astype(str).values  # Categorical as string
        return data_dict

    X_train_dict = df_to_dict(X_train)
    X_val_dict = df_to_dict(X_val)
    X_test_dict = df_to_dict(X_test)

    return X_train_dict, X_val_dict, X_test_dict, y_train_enc, y_val_enc, y_test_enc, preprocessing_layers