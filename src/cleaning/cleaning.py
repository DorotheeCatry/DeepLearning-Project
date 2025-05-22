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
    
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Original data shape: {df.shape}")
    
    
    # Conversion de la colonne 'TotalCharges' en numérique
    # Certaines valeurs sont des chaînes vides (" "), donc on utilise 'coerce' pour convertir ces cas en NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # On remplace les valeurs manquantes par 0 *uniquement* pour les clients avec 0 mois d'ancienneté
    # Un client tout juste inscrit n'a pas encore généré de facturation
    df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isna()), 'TotalCharges'] = 0
    # Conversion au type float32 pour Deep Learning (plus efficace en mémoire)
    df['TotalCharges'] = df['TotalCharges'].astype('float32')

    # Convert categorical columns to lowercase
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].str.lower() # Convert to lowercase
        df[col] = df[col].str.strip() # Remove leading/trailing whitespace
    
    # Convert 'SeniorCitizen' from 0/1 to 'no'/'yes'
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'no', 1: 'yes'})
    
    # Save cleaned data
    df.to_csv('data/cleaned_data.csv', index=False)
    print("Cleaned data saved to 'data/cleaned_data.csv'")
    
    return df

if __name__ == "__main__":
    clean_data()