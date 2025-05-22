import pandas as pd
import numpy as np
import os

def clean_data():
    """
    Clean the Telco Customer Churn dataset.
    
    This function:
    1. Loads the raw data
    2. Converts TotalCharges to float
    3. Imputes missing values
    4. Converts categorical variables to lowercase
    5. Saves the cleaned data
    
    Returns:
        pandas DataFrame with cleaned data
    """
    # Get data directory path
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '..', '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Try to find the data file
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'cleaned' not in f]
    
    if csv_files:
        input_path = os.path.join(data_dir, csv_files[0])
        print(f"Loading data from {input_path}")
    else:
        raise FileNotFoundError("No CSV data files found in data directory.")
    
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")
    
    # Convert TotalCharges to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    # Impute missing values in TotalCharges with median
    if df["TotalCharges"].isnull().sum() > 0:
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    
    # Convert categorical columns to lowercase
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].str.lower()
    
    # Convert 'SeniorCitizen' from 0/1 to 'no'/'yes'
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'no', 1: 'yes'})
    
    # Save cleaned data
    output_path = os.path.join(data_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    clean_data()