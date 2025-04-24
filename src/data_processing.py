"""
Module for loading and basic preprocessing of the store item demand data.
"""

import pandas as pd
import os
import logging
from src import config
from src.utils import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data(df: pd.DataFrame, set_index: bool = True) -> pd.DataFrame:
    """Performs basic data preparation.

    Currently sets the date as index (optional).
    Can be expanded later for cleaning, type conversion etc.

    Args:
        df: Input DataFrame (train or test).
        set_index: Whether to set the 'date' column as the index.

    Returns:
        The prepared DataFrame.
    """
    if df is None:
        return None

    df_copy = df.copy()

    if set_index:
        if 'date' in df_copy.columns:
            logging.info("Setting 'date' column as index.")
            df_copy.set_index('date', inplace=True)
        else:
            logging.warning("'date' column not found, cannot set as index.")

    # Add more preparation steps here if needed later (e.g., outlier handling)

    return df_copy

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    logging.info("Running data_processing module directly...")
    # Assume the script is run from the project root or src directory
    # Adjust path as necessary
    data_dir = '../data'
    if not os.path.exists(data_dir):
        data_dir = 'data' # Try relative to current dir if ../data doesn't exist

    train_data = load_data(config.RAW_DATA_DIR, config.TRAIN_FILENAME, config.RAW_TRAIN_COLS)
    test_data = load_data(config.RAW_DATA_DIR, config.TEST_FILENAME, config.RAW_TEST_COLS)

    if train_data is not None and test_data is not None:
        logging.info("Data loaded successfully.")
        train_prepared = prepare_data(train_data, set_index=True)
        test_prepared = prepare_data(test_data, set_index=False) # Usually don't index test set yet

        if train_prepared is not None:
            print("\n--- Prepared Train Data Head ---")
            print(train_prepared.head())
            print("\n--- Prepared Train Data Info ---")
            train_prepared.info()

        if test_prepared is not None:
            print("\n--- Prepared Test Data Head ---")
            print(test_prepared.head())
            print("\n--- Prepared Test Data Info ---")
            test_prepared.info()
    else:
        logging.error("Failed to load or prepare data.") 