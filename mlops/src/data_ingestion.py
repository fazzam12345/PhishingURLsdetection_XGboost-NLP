import os
import pandas as pd

def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    """
    Loads raw data from the specified path.
    Args:
        raw_data_path (str): Path to the raw data file.
    Returns:
        pd.DataFrame: Raw data in a pandas DataFrame.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    return df