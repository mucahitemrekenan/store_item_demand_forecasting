# apps/streamlit_app/utils.py
import streamlit as st
import pandas as pd
import joblib
import os
import logging

@st.cache_data # Keep caching for raw data loading
def load_raw_data(data_dir: str, train_filename: str):
    """Loads the raw training data from train.csv."""
    train_file = os.path.join(data_dir, train_filename)
    try:
        if not os.path.exists(train_file):
             st.error(f"Training data file not found at {train_file}.")
             return None
        train_df = pd.read_csv(train_file, parse_dates=['date'])
        if 'date' not in train_df.columns or not pd.api.types.is_datetime64_any_dtype(train_df['date']):
            st.error("Invalid 'date' column in training data.")
            return None
        return train_df
    except FileNotFoundError:
         st.error(f"Training data file not found: {train_file}")
         return None
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        logging.error(f"Error loading raw data from {train_file}: {e}")
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