# Contents for EDA tab
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Assuming DATE_FEATURES is imported where this function is called or passed implicitly
# For standalone execution or testing, you might need to define/import it here.

def render_eda_tab(historical_data, eda_feature_options, lag_feature_names):
    """Renders the EDA tab content."""
    with st.container():
        st.subheader("Exploratory Data Analysis")
        
        if historical_data.empty:
            st.warning("No historical data for this store-item in the selected date range.")
        elif not eda_feature_options:
             st.warning("No numeric features available to plot.")
        else:
            # --- Feature Plotting --- 
            st.markdown("#### Feature Over Time")
            col1, col2 = st.columns([1, 3]) # Make dropdown column narrower
            with col1:
                # Dropdown to select feature
                selected_eda_feature = st.selectbox(
                    "Select feature:", 
                    options=eda_feature_options,
                    index=eda_feature_options.index('sales') if 'sales' in eda_feature_options else 0, # Default to 'sales'
                    key='eda_feature_select'
                )
            
            # Plot the selected feature
            fig_hist = px.line(historical_data, x='date', y=selected_eda_feature, title=f"Daily {selected_eda_feature.capitalize()} History")
            fig_hist.update_layout(xaxis_title="Date", yaxis_title=selected_eda_feature.capitalize(), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Add a note about potential NaNs for generated features
            # A simple check: if it's not sales and not a known basic date feature name pattern
            # This relies on DATE_FEATURES being available/correctly defined in the calling scope
            # if selected_eda_feature != 'sales' and not any(f == selected_eda_feature for f in DATE_FEATURES):
            # More robust check might be needed depending on how DATE_FEATURES is handled.
            # Let's use the presence in lag_feature_names as a proxy for generated features.
            if selected_eda_feature in lag_feature_names or 'roll' in selected_eda_feature:
                st.caption("Note: Lag and Rolling features may contain initial NaN values.")
            
            # --- Descriptive Statistics --- 
            st.markdown(f"#### Descriptive Statistics: `{selected_eda_feature}`")
            if selected_eda_feature in historical_data.columns:
                st.dataframe(historical_data[[selected_eda_feature]].describe(), use_container_width=False)
            else:
                st.warning(f"Selected feature '{selected_eda_feature}' not found in the data.")
                
            st.divider() # Add a visual separator
            
            # --- Correlation Heatmap --- 
            st.markdown("#### Feature Correlation Heatmap")
            
            # Define default features for correlation heatmap
            default_corr_features = ['sales', 'dayofweek', 'month']
            first_lag = next((f for f in lag_feature_names if f in eda_feature_options), None)
            if first_lag: default_corr_features.append(first_lag)
            default_corr_features = [f for f in default_corr_features if f in eda_feature_options]

            selected_corr_features = st.multiselect(
                "Select features for correlation heatmap:",
                options=eda_feature_options,
                default=default_corr_features, 
                key='eda_corr_multiselect'
            )
            
            if len(selected_corr_features) >= 2:
                try:
                    # Ensure only numeric columns are used for correlation
                    numeric_selected_features = historical_data[selected_corr_features].select_dtypes(include=np.number).columns.tolist()
                    if len(numeric_selected_features) < 2:
                        st.info("Please select at least two *numeric* features for correlation.")
                    else:
                        corr_matrix = historical_data[numeric_selected_features].corr()
                        
                        fig_corr = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            aspect="auto", 
                            color_continuous_scale='RdBu_r', 
                            zmin=-1, zmax=1, 
                            title="Correlation Matrix"
                        )
                        fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate correlation heatmap: {e}")
            elif len(selected_corr_features) == 1:
                st.info("Please select at least two features to calculate correlation.")
            else:
                st.info("Select features using the dropdown above to generate a correlation heatmap.") 