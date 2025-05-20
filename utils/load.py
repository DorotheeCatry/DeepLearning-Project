import os
import pandas as pd

def load_data(filename='cleaned_datas.csv'):
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', filename)
    return pd.read_csv(data_path)