# apps/streamlit_app/sidebar.py
import streamlit as st
import pandas as pd
import os
import time
import logging

# Import necessary functions from other modules
# Ensure PYTHONPATH is set correctly or use relative imports if appropriate
try:
    from src.modeling import run_training_pipeline
    # Import helpers from the new utils module within the same package
    from src.utils import load_raw_data, load_model
except ImportError as e:
    st.sidebar.error(f"Sidebar Error: Could not import necessary functions: {e}")
    # Define dummies if needed, although sidebar might be non-functional
    def run_training_pipeline(*args, **kwargs): return None, None
    def load_raw_data(*args, **kwargs): 
        st.warning("Dummy load_raw_data used in sidebar.")
        return None
    def load_model(*args, **kwargs): 
        st.warning("Dummy load_model used in sidebar.")
        return None

def render_sidebar(RAW_DATA_DIR, MODELS_DIR, initial_forecast_horizon, train_df):
    """
    Renders the sidebar content and returns user selections.

    Args:
        RAW_DATA_DIR (str): Path to the data directory.
        MODELS_DIR (str): Path to the models directory.
        initial_forecast_horizon (int): Default value for forecast horizon.
        train_df (pd.DataFrame): The base training dataframe for populating selectors.

    Returns:
        dict: A dictionary containing user selections:
              'selected_store', 'selected_item', 'start_date', 'end_date',
              'selected_model_file', 'loaded_model', 'forecast_horizon'
    """
    st.sidebar.header("Settings")

    # --- Data Upload Section ---
    st.sidebar.header("Upload New Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload new training data (CSV format)",
        type=["csv"],
        help="Upload a CSV file with columns: date, store, item, sales. This will overwrite the existing train.csv."
    )

    if uploaded_file is not None:
        # Simplified processing logic from main app
        try:
            new_data_df = pd.read_csv(uploaded_file)
            required_cols = {'date', 'store', 'item', 'sales'}
            if required_cols.issubset(new_data_df.columns):
                new_data_df['date'] = pd.to_datetime(new_data_df['date'])
                if pd.api.types.is_numeric_dtype(new_data_df['sales']):
                    os.makedirs(RAW_DATA_DIR, exist_ok=True)
                    new_data_df.to_csv(RAW_DATA_DIR, index=False)
                    st.sidebar.success(f"Data saved to {RAW_DATA_DIR}!")
                    # Check if the imported load_raw_data has the clear method
                    if hasattr(load_raw_data, 'clear'):
                         load_raw_data.clear() # Clear cache
                    st.experimental_rerun() # Rerun to use new data
                else:
                    st.sidebar.error("Upload failed: 'sales' not numeric.")
            else:
                st.sidebar.error(f"Upload failed: Missing cols: {required_cols - set(new_data_df.columns)}")
        except Exception as e:
            st.sidebar.error(f"Error processing upload: {e}")
            logging.error(f"File upload error: {e}")


    # --- Model Training Section ---
    st.sidebar.header("Model Training")
    with st.sidebar.expander("Train New Model", expanded=False):
        retrain_full_data = st.checkbox("Retrain on full data", value=True)
        train_button = st.button("Train New Model")
        if train_button:
            # Check if necessary functions were imported correctly
            # No need to check globals if import succeeded
            with st.spinner("Training new model..."):
                training_status = st.sidebar.empty() # Place status in sidebar
                training_status.info("Starting training pipeline...")
                start_time = time.time()
                try:
                    final_model, cv_scores = run_training_pipeline(
                        data_path=RAW_DATA_DIR,
                        model_output_dir=MODELS_DIR,
                        retrain_on_full_data=retrain_full_data
                    )
                    duration = time.time() - start_time
                    if final_model and cv_scores:
                        training_status.success(f"Training done ({duration:.2f}s)! Refresh may be needed.")
                        st.sidebar.metric("CV MAE", f"{cv_scores['mae']:.4f}")
                        st.sidebar.metric("CV RMSE", f"{cv_scores['rmse']:.4f}")
                        st.sidebar.metric("CV SMAPE", f"{cv_scores['smape']:.4f}")
                        # Clear model cache to ensure newly trained model can be loaded
                        if hasattr(load_model, 'clear'):
                            load_model.clear()
                        # Consider adding st.experimental_rerun() here too if model list needs instant update
                    else:
                        training_status.error("Training failed. Check logs.")
                except Exception as e:
                    training_status.error(f"Training error: {e}")
                    logging.error(f"Training error: {e}")

    # --- Model Selection ---
    st.sidebar.header("Model Selection")
    loaded_model = None
    selected_model_file = None
    try:
        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')], reverse=True)
    except FileNotFoundError:
        st.sidebar.warning(f"Model directory '{MODELS_DIR}' not found.")
        model_files = []

    if not model_files:
        st.sidebar.warning("No trained models found. Please train a model.")
    else:
        selected_model_file = st.sidebar.selectbox(
            "Select Model", model_files, index=0, key='selected_model_file_key'
        )
        if selected_model_file:
            model_path = os.path.join(MODELS_DIR, selected_model_file)
            loaded_model = load_model(model_path)
            if loaded_model is None:
                st.sidebar.error(f"Could not load model: {selected_model_file}")

    # --- Input Controls ---
    st.sidebar.header("Analysis Parameters")
    forecast_horizon = st.sidebar.number_input(
        "Forecast horizon (days)", min_value=1, max_value=365, value=initial_forecast_horizon
    )

    # Date Range - Requires train_df
    start_date, end_date = None, None # Initialize
    if train_df is not None and not train_df.empty:
        try:
             min_date = train_df['date'].min().date()
             max_date = train_df['date'].max().date()
             start_date, end_date = st.sidebar.date_input(
                 "Historical Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
             )
        except Exception as e:
             st.sidebar.warning(f"Could not set date range: {e}")
             # Keep start_date, end_date as None or set defaults
             start_date, end_date = pd.Timestamp('today').date() - pd.Timedelta(days=30), pd.Timestamp('today').date()
    else:
        st.sidebar.warning("Cannot set date range: No training data loaded.")
        # Set defaults if no data
        start_date, end_date = pd.Timestamp('today').date() - pd.Timedelta(days=30), pd.Timestamp('today').date()

    # Store/Item Selection - Requires train_df
    selected_store, selected_item = None, None # Initialize
    if train_df is not None and not train_df.empty:
        try:
            stores = sorted(train_df['store'].unique())
            items = sorted(train_df['item'].unique())
            selected_store = st.sidebar.selectbox("Select Store", options=stores, index=0 if stores else None)
            selected_item = st.sidebar.selectbox("Select Item", options=items, index=0 if items else None)
        except Exception as e:
            st.sidebar.warning(f"Could not populate Store/Item selectors: {e}")
            # Keep selections as None
    else:
        st.sidebar.warning("Cannot select Store/Item: No training data loaded.")
        # Keep selections as None

    return {
        'selected_store': selected_store,
        'selected_item': selected_item,
        'start_date': start_date,
        'end_date': end_date,
        'selected_model_file': selected_model_file,
        'loaded_model': loaded_model,
        'forecast_horizon': forecast_horizon
    } 