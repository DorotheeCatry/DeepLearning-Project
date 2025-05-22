import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import load_data
from utils.preprocessing import preprocess
from utils.split import split_data

def main():

    # Load the data
    df = load_data()

    # Split and preprocess datas set
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_processed, X_test_processed, X_val_processed, y_test_encoding, y_train_encoded, y_val_encoding, pipeline, le = preprocess(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Transformation of dataframes in array
    X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
    X_val_dense = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
    X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
    y_train_vector = y_train_encoded.reshape(-1)
    y_val_vector = y_val_encoding.reshape(-1)
    y_test_vector = y_test_encoding.reshape(-1)
    
    pass

#A mettre à la fin
if __name__ == "__main__":
    main()