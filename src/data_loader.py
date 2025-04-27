import os
import pandas as pd

def load_raw_data(filepath):
    """
    Load raw transaction data from a CSV file.
    
    Args:
        filepath (str): Path to the raw transactions CSV file.

    Returns:
        pd.DataFrame: Loaded transaction data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def save_processed_data(df, filepath):
    """
    Save processed feature-engineered data to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Path where the CSV will be saved.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}.")

def load_processed_data(filepath):
    """
    Load processed feature-engineered data from a CSV file.
    
    Args:
        filepath (str): Path to the processed features CSV file.

    Returns:
        pd.DataFrame: Loaded feature-engineered dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df
