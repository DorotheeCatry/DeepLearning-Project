import os
import pandas as pd

def load_data(filename='cleaned_datas.csv'):
    """
    Load the cleaned dataset from the data directory.
    
    Args:
        filename: Name of the cleaned data file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', filename)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {data_path} not found.")
        print("Make sure to run the data cleaning script first.")
        return None