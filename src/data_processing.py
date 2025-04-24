"""
Module for loading and basic preprocessing of the store item demand data.
"""

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the train and test datasets from the specified path.

    Args:
        data_path: The path to the directory containing train.csv and test.csv.

    Returns:
        A tuple containing the train and test pandas DataFrames.
        Returns (None, None) if files are not found.
    """
    train_file = os.path.join(data_path, 'train.csv')
    test_file = os.path.join(data_path, 'test.csv')

    if not os.path.exists(train_file):
        logging.error(f"Train file not found at {train_file}")
        return None, None
    if not os.path.exists(test_file):
        logging.error(f"Test file not found at {test_file}")
        return None, None

    try:
        logging.info(f"Loading train data from {train_file}...")
        train_df = pd.read_csv(train_file, parse_dates=['date'])
        logging.info(f"Train data loaded successfully. Shape: {train_df.shape}")

        logging.info(f"Loading test data from {test_file}...")
        test_df = pd.read_csv(test_file, parse_dates=['date'])
        logging.info(f"Test data loaded successfully. Shape: {test_df.shape}")

        # Basic Validation
        if 'date' not in train_df.columns or 'store' not in train_df.columns or \
           'item' not in train_df.columns or 'sales' not in train_df.columns:
            logging.error("Train data missing required columns (date, store, item, sales).")
            return None, None
        if 'date' not in test_df.columns or 'store' not in test_df.columns or \
           'item' not in test_df.columns or 'id' not in test_df.columns:
             logging.error("Test data missing required columns (date, store, item, id).")
             return None, None

        # Convert Store and Item to category type for efficiency
        # train_df['store'] = train_df['store'].astype('category')
        # train_df['item'] = train_df['item'].astype('category')
        # test_df['store'] = test_df['store'].astype('category')
        # test_df['item'] = test_df['item'].astype('category')
        # Note: Consider if this is beneficial later depending on feature engineering

        return train_df, test_df

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

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

    train_data, test_data = load_data(data_dir)

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