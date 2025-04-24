"""
Streamlit application for Store Item Demand Forecasting.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
# import joblib # No longer needed here directly
import logging
import time

# --- Project Setup ---
# Add project root to path if necessary (running `streamlit run apps/streamlit_app/streamlit_dashboard.py` from root should work)
# Alternatively, install the src package

# --- Imports ---
try:
    # Core functionalities from src and config
    from src.feature_engineering import create_date_features, create_lag_features, create_rolling_features
    # Import all necessary config variables from src.config
    from src import config 

    # App components
    from apps.streamlit_app.eda_tab import render_eda_tab
    from apps.streamlit_app.features_tab import render_features_tab
    from apps.streamlit_app.model_info_tab import render_model_info_tab
    from apps.streamlit_app.forecast_tab import render_forecast_tab
    from apps.streamlit_app.sidebar import render_sidebar
    from src.utils import load_raw_data, load_model

except ImportError as e:
    st.error(f"Fatal Error: Could not import necessary modules: {e}. "
             f"Ensure the app is run from the project root (`streamlit run apps/streamlit_app/streamlit_dashboard.py`) "
             f"or PYTHONPATH is correctly set.")
    st.exception(e) # Show full traceback in Streamlit for debugging
    st.stop()

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Removed Helper Functions (Moved to utils.py) --- 
# load_raw_data and load_model are now in apps/streamlit_app/utils.py

# --- Main Application Logic ---
def main():
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide", page_title=config.APP_TITLE)
    st.title(config.APP_TITLE)
    st.markdown("Predict future sales for specific store-item combinations.")

    # --- Load Base Data --- 
    # Use config for data directory
    train_df = load_raw_data(config.RAW_DATA_DIR, config.TRAIN_FILENAME)
    if train_df is None:
        st.error(f"Failed to load training data from {os.path.join(config.RAW_DATA_DIR, config.TRAIN_FILENAME)}. Please check the path and file.")
        st.stop()

    # --- Render Sidebar and Get Selections ---
    sidebar_selections = render_sidebar(
        config.RAW_DATA_DIR, 
        config.MODEL_OUTPUT_DIR, # Use config
        config.FORECAST_HORIZON, # Use config for default
        train_df
    )

    # Extract selections
    selected_store = sidebar_selections['selected_store']
    selected_item = sidebar_selections['selected_item']
    start_date = sidebar_selections['start_date']
    end_date = sidebar_selections['end_date']
    model = sidebar_selections['loaded_model']
    current_forecast_horizon = sidebar_selections['forecast_horizon']

    # --- Validation after Sidebar ---
    if selected_store is None or selected_item is None:
        st.warning("Please select a Store and Item in the sidebar.")
        st.stop()
    if start_date is None or end_date is None:
        st.warning("Please select a valid date range in the sidebar.")
        st.stop()

    # --- Prepare Data for Tabs --- 
    st.header(f"Store {selected_store} – Item {selected_item}")

    # Filter based on sidebar selections
    historical_data_filtered = pd.DataFrame()
    try:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        historical_data_filtered = train_df[
            (train_df[config.GROUP_COLS[0]] == selected_store) & # Use config for group cols
            (train_df[config.GROUP_COLS[1]] == selected_item) & 
            (train_df[config.DATE_COL] >= start_date_dt) & # Use config for date col
            (train_df[config.DATE_COL] <= end_date_dt)
        ].copy()
        historical_data_filtered = historical_data_filtered.sort_values(config.DATE_COL)
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        st.stop()

    # Generate features needed for EDA/Features tabs (on filtered data)
    historical_data_with_features = historical_data_filtered.copy()
    eda_feature_options = [config.TARGET_COL] 
    lag_feature_names_eda = []
    rolling_feature_names_eda = []
    date_feature_names_eda = []
    
    if not historical_data_with_features.empty:
        try:
            if config.CREATE_DATE_FEATURES:
                historical_data_with_features = create_date_features(
                    historical_data_with_features, 
                    date_col=config.DATE_COL, 
                    include_cyclical=config.INCLUDE_CYCLICAL_DATE_FEATURES, 
                    add_holidays=config.ADD_HOLIDAY_FEATURES
                )
                # Infer date features added (replace manual list)
                # This is less robust than having create_date_features return names
                potential_date_feats = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'is_month_end', 'is_quarter_end']
                if config.INCLUDE_CYCLICAL_DATE_FEATURES: potential_date_feats.extend(['month_sin', 'month_cos', 'dow_sin', 'dow_cos'])
                if config.ADD_HOLIDAY_FEATURES and 'is_holiday' in historical_data_with_features.columns: potential_date_feats.append('is_holiday')
                date_feature_names_eda = [f for f in potential_date_feats if f in historical_data_with_features.columns and pd.api.types.is_numeric_dtype(historical_data_with_features[f])]
                eda_feature_options.extend(date_feature_names_eda)

            if config.CREATE_LAG_FEATURES:
                historical_data_with_features = create_lag_features(
                    historical_data_with_features, 
                    group_cols=config.GROUP_COLS, 
                    target_col=config.TARGET_COL, 
                    lags=config.LAG_FEATURES_LIST, 
                    fill_method=config.LAG_FILL_METHOD
                )
                lag_feature_names_eda = [f'{config.TARGET_COL}_lag_{l}' for l in config.LAG_FEATURES_LIST]
                lag_options = [f for f in lag_feature_names_eda if f in historical_data_with_features.columns and pd.api.types.is_numeric_dtype(historical_data_with_features[f])]
                eda_feature_options.extend(lag_options)

            if config.CREATE_ROLLING_FEATURES:
                historical_data_with_features = create_rolling_features(
                    historical_data_with_features, 
                    group_cols=config.GROUP_COLS, 
                    target_col=config.TARGET_COL, 
                    windows=config.ROLLING_WINDOWS, 
                    aggs=config.ROLLING_AGGS, 
                    fill_method=config.ROLLING_FILL_METHOD
                )
                rolling_feature_names_eda = [f'{config.TARGET_COL}_roll_{agg}_{w}d' for w in config.ROLLING_WINDOWS for agg in config.ROLLING_AGGS]
                rolling_options = [f for f in rolling_feature_names_eda if f in historical_data_with_features.columns and pd.api.types.is_numeric_dtype(historical_data_with_features[f])]
                eda_feature_options.extend(rolling_options)
            
            eda_feature_options = sorted(list(set(eda_feature_options)))
            
        except Exception as e:
            st.warning(f"Could not generate features for display: {e}")
            eda_feature_options = [config.TARGET_COL] # Minimal fallback

    # --- Main Content Area (Tabs) --- 
    tab_labels = ["EDA", "Features", "Model Info", "Forecast"]
    if 'main_tabs_radio' not in st.session_state:
        st.session_state.main_tabs_radio = "EDA"

    st.radio(
        "Select View:", 
        tab_labels, 
        key="main_tabs_radio", 
        horizontal=True,
        label_visibility="collapsed"
    )
    active_view = st.session_state.main_tabs_radio
    
    if active_view == "EDA":
        render_eda_tab(
            historical_data=historical_data_with_features, 
            eda_feature_options=eda_feature_options,
            lag_feature_names=lag_feature_names_eda 
        )
    elif active_view == "Features":
        render_features_tab(
            historical_data=historical_data_with_features, 
            lag_feature_names=lag_feature_names_eda,
            rolling_feature_names=rolling_feature_names_eda
        )
    elif active_view == "Model Info":
        render_model_info_tab(model=model) 
    elif active_view == "Forecast":
        # Pass the potentially adjusted forecast horizon from the sidebar
        render_forecast_tab(
            model=model, 
            historical_data=historical_data_filtered, # Pass only necessary historical data
            selected_store=selected_store,
            selected_item=selected_item,
            FORECAST_HORIZON=current_forecast_horizon # Use value from sidebar
        )

if __name__ == "__main__":
    main()




