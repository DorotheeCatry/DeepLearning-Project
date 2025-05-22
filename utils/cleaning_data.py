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
    os.makedirs(data_dir, exist_ok=True)
    
    # Try to find the data file
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'cleaned' not in f]
    
    if csv_files:
        input_path = os.path.join(data_dir, csv_files[0])
        print(f"Loading data from {input_path}")
    else:
        # Check if data is in the current directory
        current_dir_csv = [f for f in os.listdir('..') if f.endswith('.csv')]
        if current_dir_csv:
            input_path = os.path.join('..', current_dir_csv[0])
            print(f"Loading data from {input_path}")
        else:
            raise FileNotFoundError("No CSV data files found in data directory or current directory.")
    
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
    
    # Convert 'SeniorCitizen' from 0/1 to 'no'/'yes' for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'no', 1: 'yes'})
    
    # Check for missing values after cleaning
    missing_after = df.isnull().sum()
    print("Missing values after cleaning:")
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "No missing values")
    
    # Save df into a cleaned csv
    output_path = os.path.join(data_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    # Data quality checks
    print("\nData Quality Checks:")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    print(f"Number of unique customers: {df['customerID'].nunique()}")
    
    # Check for outliers in numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        if outliers > 0:
            print(f"Column {col} has {outliers} outliers")
    
    return df

if __name__ == "__main__":
    clean_data()