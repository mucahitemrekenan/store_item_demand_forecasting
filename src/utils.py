# apps/streamlit_app/utils.py
import streamlit as st
import pandas as pd
import joblib
import os
import logging


@st.cache_data
def load_data(data_dir: str, file_name: str, cols_to_check: list[str]) -> pd.DataFrame:
    """Loads the dataset from the specified path.

    Args:
        data_dir: The path to the directory containing csv file.

    Returns:
        DataFrame
        Returns None if files are not found.
    """
    file_name = os.path.join(data_dir, file_name)

    if not os.path.exists(file_name):
        logging.error(f"File not found at {file_name}")
        return None

    try:
        logging.info(f"Loading data from {file_name}...")
        df = pd.read_csv(file_name, parse_dates=['date'])
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        # Basic Validation
        if not all(col in df.columns for col in cols_to_check):
            logging.error("Data missing required columns: {cols_to_check}.")
            return None
        if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
            st.error("Invalid 'date' column in data.")
            return None
        return df
    except FileNotFoundError:
         st.error(f"Data file not found: {file_name}")
         return None
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        logging.error(f"Error loading raw data {file_name}: {e}")
        return None

@st.cache_data # Keep model loading cached here
def load_model(model_path: str):
    """Loads the pre-trained model."""
    if not model_path or not isinstance(model_path, str):
        return None
    try:
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found at {model_path}")
            return None
        model = joblib.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None 

