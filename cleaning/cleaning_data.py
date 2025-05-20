# Import
import pandas as pd
import numpy as np
import os

# Dynamic current file path
current_dir = os.path.dirname(__file__)

# Dynamic path to data
data_dir = os.path.join(current_dir, '..', 'data')

# Get the data
input_path = os.path.join(data_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(input_path)

# Cleaning Tasks

# Transformation of the TotalCharges column in float
df["TotalCharges"] = df["TotalCharges"].str.strip().replace(",", ".", regex=True).replace("", np.nan).astype(float)

# Transformation of all object columns in lowercase
columns_to_lower = ['gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn']
for c in columns_to_lower:

    df[c]=df[c].apply(lambda x: x.lower())

# Save df into a cleaned csv
output_path = os.path.join(data_dir, 'cleaned_datas.csv')
df.to_csv(output_path, index=False)