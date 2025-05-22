import pandas as pd

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
    
    df = pd.read_csv('data/Telco-Customer-Churn.csv')
    print(f"Original data shape: {df.shape}")
    
    # Convert the 'TotalCharges' column to numeric
    # Some values are empty strings (" "), so we use 'coerce' to convert these cases to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Replace missing values with 0 *only* for customers with 0 months of tenure
    # A newly registered customer has not generated any billing yet
    df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isna()), 'TotalCharges'] = 0
    # Convert to float32 type for Deep Learning (more memory efficient)
    df['TotalCharges'] = df['TotalCharges'].astype('float32')

    # Convert categorical columns to lowercase
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].str.lower() # Convert to lowercase
        df[col] = df[col].str.strip() # Remove leading/trailing whitespace
    
    # Convert 'SeniorCitizen' from 0/1 to 'no'/'yes'
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'no', 1: 'yes'})
    
    # Save cleaned data
    df.to_csv('data/Telco-Customer-Chrun_cleaned.csv', index=False)
    print("Cleaned data saved to 'data/Telco-Customer-Chrun_cleaned.csv'")
    
    return df

if __name__ == "__main__":
    clean_data()
