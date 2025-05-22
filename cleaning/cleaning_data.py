# Import
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
    # Dynamic current file path
    current_dir = os.path.dirname(__file__)
    
    # Dynamic path to data
    data_dir = os.path.join(current_dir, '..', 'data')
    
    # Get the data
    input_path = os.path.join(data_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    
    # Check for missing values before cleaning
    missing_before = df.isnull().sum()
    print("Missing values before cleaning:")
    print(missing_before[missing_before > 0] if missing_before.sum() > 0 else "No missing values")
    
    # Transformation of the TotalCharges column to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    # Impute missing values in TotalCharges with the median
    missing_total_charges = df["TotalCharges"].isnull().sum()
    if missing_total_charges > 0:
        print(f"Imputing {missing_total_charges} missing values in TotalCharges with median")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    
    # Transformation of all object columns to lowercase
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].str.lower()
    
    # Check for missing values after cleaning
    missing_after = df.isnull().sum()
    print("Missing values after cleaning:")
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "No missing values")
    
    # Save df into a cleaned csv
    output_path = os.path.join(data_dir, 'cleaned_datas.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    clean_data()