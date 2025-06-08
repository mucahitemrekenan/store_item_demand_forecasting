# Contents for Features tab
import streamlit as st
import pandas as pd

def render_features_tab(historical_data, lag_feature_names, rolling_feature_names):
    """Renders the Features tab content."""
    with st.container():
        st.subheader("Feature Engineering Preview")
        st.divider()
        if historical_data.empty:
            st.warning("No historical data to generate features from.")
        else:
            # Display features based on the *filtered* historical data (already calculated)
            # Select relevant columns to display (e.g., date, sales, and generated features)
            cols_to_show = ['date', 'sales'] + lag_feature_names + rolling_feature_names
            # Ensure columns actually exist in the dataframe before trying to select them
            cols_to_show = [col for col in cols_to_show if col in historical_data.columns] 
            
            if not cols_to_show:
                st.warning("No features generated or found to display.")
            else:
                st.dataframe(historical_data[cols_to_show].tail(10), use_container_width=True) 