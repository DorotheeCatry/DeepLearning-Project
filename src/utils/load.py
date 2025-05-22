import os
import pandas as pd

def load_data(filename=None):
    """
    Load the dataset from the data directory.
    
    Args:
        filename: Name of the data file (if None, will look for CSV files in data directory)
        
    Returns:
        pandas DataFrame with the loaded data
    """
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '..', '..', 'data')
    
    # If no filename is provided, look for CSV files in the data directory
    if filename is None:
        # Check if cleaned data exists
        cleaned_path = os.path.join(data_dir, 'cleaned_data.csv')
        if os.path.exists(cleaned_path):
            print(f"Loading cleaned data from {cleaned_path}")
            return pd.read_csv(cleaned_path)
        
        # Look for any CSV files in the data directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            filename = csv_files[0]
            print(f"Found data file: {filename}")
        else:
            # If no CSV files found, check if data is in the current directory
            current_dir_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
            if current_dir_csv:
                filename = current_dir_csv[0]
                data_path = filename
                print(f"Found data file in current directory: {filename}")
                df = pd.read_csv(data_path)
                
                # Save to data directory
                os.makedirs(data_dir, exist_ok=True)
                output_path = os.path.join(data_dir, filename)
                df.to_csv(output_path, index=False)
                print(f"Saved data to {output_path}")
                return df
            else:
                raise FileNotFoundError("No CSV data files found in data directory or current directory.")
    
    data_path = os.path.join(data_dir, filename)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        # If the specific file is not found, try to find any CSV file
        try:
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                filename = csv_files[0]
                data_path = os.path.join(data_dir, filename)
                print(f"Specified file not found. Loading {data_path} instead.")
                df = pd.read_csv(data_path)
                print(f"Dataset shape: {df.shape}")
                return df
            else:
                # If no CSV files in data directory, check current directory
                current_dir_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
                if current_dir_csv:
                    filename = current_dir_csv[0]
                    print(f"Loading data from current directory: {filename}")
                    df = pd.read_csv(filename)
                    
                    # Save to data directory
                    os.makedirs(data_dir, exist_ok=True)
                    output_path = os.path.join(data_dir, filename)
                    df.to_csv(output_path, index=False)
                    print(f"Saved data to {output_path}")
                    return df
                else:
                    raise FileNotFoundError("No CSV data files found in data directory or current directory.")
        except Exception as e:
            print(f"Error: {e}")
            return None