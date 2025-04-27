
import os
import pandas as pd
from datetime import datetime

def timestamp_now():
    """
    Generate a current timestamp string (YYYYMMDD_HHMMSS).
    
    Returns:
        str: Timestamp formatted as YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_folder_if_not_exists(path):
    """
    Create the folder if it does not exist.
    
    Args:
        path (str): Directory path to create.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")

def save_dataframe(df, filepath):
    """
    Save a pandas DataFrame to a CSV file, ensuring directory exists.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Destination file path.

    Returns:
        None
    """
    create_folder_if_not_exists(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
