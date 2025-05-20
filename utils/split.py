# Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Dynamic current file path
current_dir = os.path.dirname(__file__)

# Dynamic path to data
data_dir = os.path.join(current_dir, '..', 'data')

# Get the data
input_path = os.path.join(data_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(input_path)

# Splitting Tasks

# Split the target and features
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split of all the data in train (80%) and test (20%)
X_train_0, X_test, y_train_0, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split of the train set to keep 20% for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_0, y_train_0, test_size=0.2, random_state=42, stratify=y_train_0
)

print(X_train.shape, X_test.shape, X_val.shape, X_test.shape, y_val.shape, y_test.shape)




# Function
def split_data(df, target='Churn', test_size=0.2, val_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, stratify=y_train_full, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test