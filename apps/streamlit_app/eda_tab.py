# Contents for EDA tab
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.utils import detect_outliers_iqr
# Assuming DATE_FEATURES is imported where this function is called or passed implicitly
# For standalone execution or testing, you might need to define/import it here.


def render_eda_tab(historical_data, eda_feature_options, lag_feature_names):
    """Renders the EDA tab content."""
    
    # Initialize session state for modified data if not exists
    if 'modified_historical_data' not in st.session_state:
        st.session_state.modified_historical_data = historical_data.copy()
    
    with st.container():
        st.subheader("Exploratory Data Analysis")
        st.divider()
        if st.session_state.modified_historical_data.empty:
            st.warning("No historical data for this store-item in the selected date range.")
        elif not eda_feature_options:
             st.warning("No numeric features available to plot.")
        else:
            # --- Feature Selection at Top Level ---
            st.markdown("#### Select Feature for Analysis")
            selected_eda_feature = st.selectbox(
                "Choose a feature to analyze:", 
                options=eda_feature_options,
                index=eda_feature_options.index('sales') if 'sales' in eda_feature_options else 0,
                key='eda_feature_select'
            )

            if selected_eda_feature in lag_feature_names or 'roll' in selected_eda_feature:
                st.caption("Note: Lag and Rolling features may contain initial NaN values.")

            st.divider()

            # --- Descriptive Statistics --- 
            st.markdown(f"#### Descriptive Statistics: `{selected_eda_feature}`")
            if selected_eda_feature in st.session_state.modified_historical_data.columns:
                st.dataframe(st.session_state.modified_historical_data[[selected_eda_feature]].describe(), use_container_width=False)
            else:
                st.warning(f"Selected feature '{selected_eda_feature}' not found in the data.")
            
            st.divider()

            # --- Missing Value Analysis for Selected Feature ---
            st.markdown("#### Missing Value Analysis")
            
            # Calculate missing value statistics for selected feature
            missing_count = st.session_state.modified_historical_data[selected_eda_feature].isnull().sum()
            missing_percentage = (missing_count / len(st.session_state.modified_historical_data) * 100).round(2)
            
            # Display missing value statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing Values", missing_count)
            with col2:
                st.metric("Missing Percentage", f"{missing_percentage}%")
            
            # Missing value handling options
            if missing_count > 0:
                st.markdown("##### Handle Missing Values")
                col1, col2 = st.columns([1, 2])
                with col1:
                    imputation_method = st.selectbox(
                        "Select imputation method:",
                        options=['Remove', 'Mean', 'Median', 'Mode', 'Forward Fill', 'Backward Fill'],
                        key='imputation_method'
                    )
                
                if st.button("Apply Missing Value Handling"):
                    if imputation_method == 'Remove':
                        st.session_state.modified_historical_data = st.session_state.modified_historical_data.dropna(subset=[selected_eda_feature])
                        st.success(f"Rows with missing values in {selected_eda_feature} removed.")
                        st.rerun()
                    else:
                        if pd.api.types.is_numeric_dtype(st.session_state.modified_historical_data[selected_eda_feature]):
                            working_data_copy = st.session_state.modified_historical_data.copy()
                            if imputation_method == 'Mean':
                                working_data_copy[selected_eda_feature].fillna(working_data_copy[selected_eda_feature].mean(), inplace=True)
                            elif imputation_method == 'Median':
                                working_data_copy[selected_eda_feature].fillna(working_data_copy[selected_eda_feature].median(), inplace=True)
                            elif imputation_method == 'Mode':
                                working_data_copy[selected_eda_feature].fillna(working_data_copy[selected_eda_feature].mode()[0], inplace=True)
                            elif imputation_method == 'Forward Fill':
                                working_data_copy[selected_eda_feature].fillna(method='ffill', inplace=True)
                            elif imputation_method == 'Backward Fill':
                                working_data_copy[selected_eda_feature].fillna(method='bfill', inplace=True)
                            st.session_state.modified_historical_data = working_data_copy
                            st.success(f"Missing values in {selected_eda_feature} handled using {imputation_method}.")
                            st.rerun()
                        else:
                            st.error(f"Cannot apply numeric imputation to non-numeric feature: {selected_eda_feature}")

            st.divider()

            # --- Outlier Detection for Selected Feature ---
            if pd.api.types.is_numeric_dtype(st.session_state.modified_historical_data[selected_eda_feature]):
                st.markdown("#### Outlier Detection (IQR Method)")
                
                # Detect outliers for selected feature (using the current modified data)
                outlier_mask, lower_bound, upper_bound = detect_outliers_iqr(st.session_state.modified_historical_data[selected_eda_feature])
                outlier_count = outlier_mask.sum()
                outlier_percentage = (outlier_count / len(st.session_state.modified_historical_data) * 100).round(2)
                
                # Display outlier statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Outliers", outlier_count)
                with col2:
                    st.metric("Percentage of Outliers", f"{outlier_percentage}%")
                
                # Create box plot with outliers
                fig_box = px.box(st.session_state.modified_historical_data, y=selected_eda_feature, 
                               title=f"Box Plot with Outliers: {selected_eda_feature}")
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Create scatter plot highlighting outliers
                fig_scatter = px.scatter(st.session_state.modified_historical_data, x='date', y=selected_eda_feature,
                                       color=outlier_mask,
                                       color_discrete_map={True: 'red', False: 'blue'},
                                       title=f"Outliers Over Time: {selected_eda_feature}")
                fig_scatter.add_hline(y=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper Bound")
                fig_scatter.add_hline(y=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower Bound")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Option to handle outliers
                if outlier_count > 0:
                    st.markdown("##### Handle Outliers")
                    outlier_handling = st.selectbox(
                        "Select outlier handling method:",
                        options=['None', 'Remove', 'Cap at IQR bounds'],
                        key='outlier_handling'
                    )
                    
                    if outlier_handling != 'None' and st.button("Apply Outlier Handling"):
                        working_data_copy = st.session_state.modified_historical_data.copy()
                        if outlier_handling == 'Remove':
                            working_data_copy = working_data_copy[~outlier_mask]
                            st.session_state.modified_historical_data = working_data_copy
                            st.success(f"Removed {outlier_count} outliers from {selected_eda_feature}.")
                            st.rerun()
                        elif outlier_handling == 'Cap at IQR bounds':
                            working_data_copy.loc[working_data_copy[selected_eda_feature] > upper_bound, selected_eda_feature] = upper_bound
                            working_data_copy.loc[working_data_copy[selected_eda_feature] < lower_bound, selected_eda_feature] = lower_bound
                            st.session_state.modified_historical_data = working_data_copy
                            st.success(f"Outliers in {selected_eda_feature} capped at IQR bounds.")
                            st.rerun()

            st.divider()

            # --- Feature Plot --- 
            st.markdown("#### Feature Over Time")
            fig_hist = px.line(st.session_state.modified_historical_data, x='date', y=selected_eda_feature, 
                             title=f"Daily {selected_eda_feature.capitalize()} History")
            fig_hist.update_layout(xaxis_title="Date", yaxis_title=selected_eda_feature.capitalize(), 
                                 margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.divider()
            
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
                    numeric_selected_features = st.session_state.modified_historical_data[selected_corr_features].select_dtypes(include=np.number).columns.tolist()
                    if len(numeric_selected_features) < 2:
                        st.info("Please select at least two *numeric* features for correlation.")
                    else:
                        corr_matrix = st.session_state.modified_historical_data[numeric_selected_features].corr()
                        
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
                
            # --- Data Reset Option ---
            st.divider()
            st.markdown("#### Reset Data")
            if st.button("Reset to Original Data", help="Reset all modifications and return to original dataset"):
                st.session_state.modified_historical_data = historical_data.copy()
                st.success("Data reset to original state.")
                st.rerun() 