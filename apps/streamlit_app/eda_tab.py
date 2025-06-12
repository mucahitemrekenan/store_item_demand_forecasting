# Contents for EDA tab
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.utils import detect_outliers_iqr, perform_seasonality_decomposition, perform_stationarity_tests, calculate_advanced_statistics
# Assuming DATE_FEATURES is imported where this function is called or passed implicitly
# For standalone execution or testing, you might need to define/import it here.


def render_eda_tab(historical_data, eda_feature_options, lag_feature_names):
    """Renders the enhanced EDA tab content."""
    
    # Initialize session state for modified data if not exists
    if 'modified_historical_data' not in st.session_state:
        st.session_state.modified_historical_data = historical_data.copy()
    
    # Sidebar for EDA controls
    with st.sidebar:
        st.markdown("### 🔍 EDA Controls")
        
        # Analysis mode selection
        analysis_mode = st.selectbox(
            "Analysis Mode:",
            ["Quick Overview", "Detailed Analysis", "Statistical Deep Dive", "Comparative Analysis"],
            key='eda_analysis_mode'
        )
        
        # Feature selection
        if eda_feature_options:
            selected_eda_feature = st.selectbox(
                "Primary Feature:", 
                options=eda_feature_options,
                index=eda_feature_options.index('sales') if 'sales' in eda_feature_options else 0,
                key='eda_feature_select'
            )
        else:
            selected_eda_feature = None
            
        # Date range for analysis
        if 'date' in st.session_state.modified_historical_data.columns:
            date_range = st.date_input(
                "Analysis Date Range:",
                value=[
                    st.session_state.modified_historical_data['date'].min(),
                    st.session_state.modified_historical_data['date'].max()
                ],
                key='eda_date_range'
            )

    with st.container():
        st.title("🔍 Exploratory Data Analysis")
        st.markdown("**Comprehensive data exploration and analysis dashboard**")
        
        # Check if data is available
        if st.session_state.modified_historical_data.empty:
            st.warning("No historical data for this store-item in the selected date range.")
            return
        elif not eda_feature_options:
            st.warning("No numeric features available to plot.")
            return
        
        # Filter data by date range if provided
        filtered_data = st.session_state.modified_historical_data.copy()
        if 'date' in filtered_data.columns and len(date_range) == 2:
            mask = (filtered_data['date'] >= pd.to_datetime(date_range[0])) & \
                   (filtered_data['date'] <= pd.to_datetime(date_range[1]))
            filtered_data = filtered_data[mask]
        
        # Dataset Overview
        st.markdown("## 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(filtered_data))
        with col2:
            st.metric("Features", len(filtered_data.columns))
        with col3:
            missing_pct = (filtered_data.isnull().sum().sum() / (len(filtered_data) * len(filtered_data.columns)) * 100)
            st.metric("Missing Values %", f"{missing_pct:.2f}%")
        with col4:
            if 'date' in filtered_data.columns:
                date_range_days = (filtered_data['date'].max() - filtered_data['date'].min()).days
                st.metric("Date Range (days)", date_range_days)
        
        st.divider()
        
        # Quick Data Profile
        if st.expander("📋 Quick Data Profile", expanded=(analysis_mode == "Quick Overview")):
            st.markdown("### Data Types & Missing Values")
            
            profile_data = []
            for col in filtered_data.columns:
                profile_data.append({
                    'Column': col,
                    'Data Type': str(filtered_data[col].dtype),
                    'Non-Null Count': filtered_data[col].count(),
                    'Null Count': filtered_data[col].isnull().sum(),
                    'Null %': f"{(filtered_data[col].isnull().sum() / len(filtered_data) * 100):.2f}%",
                    'Unique Values': filtered_data[col].nunique(),
                    'Memory Usage': f"{filtered_data[col].memory_usage(deep=True) / 1024:.2f} KB"
                })
            
            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True)
        
        # Feature Analysis Section
        if selected_eda_feature:
            st.markdown(f"## 🎯 Feature Analysis: `{selected_eda_feature}`")
            
            # Advanced Statistics
            with st.expander("📈 Advanced Statistical Analysis", expanded=(analysis_mode in ["Detailed Analysis", "Statistical Deep Dive"])):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Descriptive Statistics")
                    basic_stats = filtered_data[[selected_eda_feature]].describe()
                    st.dataframe(basic_stats, use_container_width=True)
                    
                    # Advanced statistics
                    adv_stats = calculate_advanced_statistics(filtered_data[selected_eda_feature])
                    
                    st.markdown("### Distribution Properties")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Skewness", f"{adv_stats['skewness']:.4f}")
                        st.metric("Kurtosis", f"{adv_stats['kurtosis']:.4f}")
                    with col_b:
                        st.metric("Normality (Shapiro-Wilk)", 
                                f"p={adv_stats['shapiro_p']:.4f}" if not np.isnan(adv_stats['shapiro_p']) else "N/A")
                        st.write("✅ Normal" if adv_stats['is_normal'] else "❌ Not Normal")
                
                with col2:
                    st.markdown("### Distribution Visualization")
                    
                    # Create distribution plots
                    fig_dist = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot'],
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Histogram with KDE
                    fig_dist.add_trace(
                        go.Histogram(x=filtered_data[selected_eda_feature], 
                                   name='Histogram', nbinsx=30, opacity=0.7),
                        row=1, col=1
                    )
                    
                    # Box plot
                    fig_dist.add_trace(
                        go.Box(y=filtered_data[selected_eda_feature], name='Box Plot'),
                        row=1, col=2
                    )
                    
                    # Q-Q plot
                    from scipy.stats import probplot
                    qq_data = probplot(filtered_data[selected_eda_feature].dropna(), dist="norm")
                    fig_dist.add_trace(
                        go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                                 mode='markers', name='Q-Q Plot'),
                        row=2, col=1
                    )
                    
                    # Violin plot
                    fig_dist.add_trace(
                        go.Violin(y=filtered_data[selected_eda_feature], name='Violin Plot'),
                        row=2, col=2
                    )
                    
                    fig_dist.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Time Series Analysis
            if 'date' in filtered_data.columns:
                with st.expander("📅 Time Series Analysis", expanded=(analysis_mode in ["Detailed Analysis", "Statistical Deep Dive"])):
                    
                    # Basic time series plot
                    st.markdown("### Time Series Visualization")
                    fig_ts = px.line(filtered_data, x='date', y=selected_eda_feature,
                                   title=f"{selected_eda_feature.capitalize()} Over Time")
                    
                    # Add rolling averages
                    if len(filtered_data) > 7:
                        filtered_data_copy = filtered_data.copy()
                        filtered_data_copy['rolling_7'] = filtered_data_copy[selected_eda_feature].rolling(7).mean()
                        filtered_data_copy['rolling_30'] = filtered_data_copy[selected_eda_feature].rolling(30).mean()
                        
                        fig_ts.add_trace(go.Scatter(
                            x=filtered_data_copy['date'], 
                            y=filtered_data_copy['rolling_7'],
                            name='7-day MA', line=dict(dash='dash')
                        ))
                        
                        if len(filtered_data) > 30:
                            fig_ts.add_trace(go.Scatter(
                                x=filtered_data_copy['date'], 
                                y=filtered_data_copy['rolling_30'],
                                name='30-day MA', line=dict(dash='dot')
                            ))
                    
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    # Seasonality Decomposition
                    st.markdown("### Seasonality Decomposition")
                    decomp_results = perform_seasonality_decomposition(
                        filtered_data[[selected_eda_feature, 'date']], 'date'
                    )
                    
                    if decomp_results:
                        fig_decomp = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                            shared_xaxes=True,
                            vertical_spacing=0.05
                        )
                        
                        for i, (component, data) in enumerate(decomp_results.items(), 1):
                            fig_decomp.add_trace(
                                go.Scatter(x=data.index, y=data.values, 
                                         name=component.capitalize(), line=dict(width=1)),
                                row=i, col=1
                            )
                        
                        fig_decomp.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    # Stationarity Tests
                    if analysis_mode == "Statistical Deep Dive":
                        st.markdown("### Stationarity Analysis")
                        adf_stats, kpss_stats = perform_stationarity_tests(filtered_data[selected_eda_feature])
                        
                        if adf_stats and kpss_stats:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Augmented Dickey-Fuller Test**")
                                st.write(f"Test Statistic: {adf_stats['test_statistic']:.4f}")
                                st.write(f"P-value: {adf_stats['p_value']:.4f}")
                                st.write("✅ Stationary" if adf_stats['is_stationary'] else "❌ Non-Stationary")
                            
                            with col2:
                                st.markdown("**KPSS Test**")
                                st.write(f"Test Statistic: {kpss_stats['test_statistic']:.4f}")
                                st.write(f"P-value: {kpss_stats['p_value']:.4f}")
                                st.write("✅ Stationary" if kpss_stats['is_stationary'] else "❌ Non-Stationary")
            
            # Missing Value Analysis (Enhanced)
            with st.expander("🔍 Missing Value Analysis", expanded=False):
                st.markdown("### Missing Value Analysis")
                
                missing_count = filtered_data[selected_eda_feature].isnull().sum()
                missing_percentage = (missing_count / len(filtered_data) * 100).round(2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Missing Values", missing_count)
                with col2:
                    st.metric("Missing Percentage", f"{missing_percentage}%")
                with col3:
                    st.metric("Complete Cases", len(filtered_data) - missing_count)
                
                # Missing value pattern
                if missing_count > 0:
                    st.markdown("#### Missing Value Pattern")
                    missing_mask = filtered_data[selected_eda_feature].isnull()
                    
                    if 'date' in filtered_data.columns:
                        fig_missing = px.scatter(
                            filtered_data, x='date', y=missing_mask.astype(int),
                            title="Missing Value Pattern Over Time",
                            labels={'y': 'Missing (1) / Present (0)'}
                        )
                        st.plotly_chart(fig_missing, use_container_width=True)
                    
                    # Missing value handling
                    st.markdown("#### Handle Missing Values")
                    imputation_method = st.selectbox(
                        "Select imputation method:",
                        options=['Remove', 'Mean', 'Median', 'Mode', 'Forward Fill', 'Backward Fill', 'Interpolate'],
                        key='imputation_method_enhanced'
                    )
                    
                    if st.button("Apply Missing Value Handling", key='apply_missing_enhanced'):
                        working_data_copy = st.session_state.modified_historical_data.copy()
                        
                        if imputation_method == 'Remove':
                            working_data_copy = working_data_copy.dropna(subset=[selected_eda_feature])
                        elif imputation_method == 'Interpolate':
                            working_data_copy[selected_eda_feature] = working_data_copy[selected_eda_feature].interpolate()
                        else:
                            # Apply other imputation methods
                            if pd.api.types.is_numeric_dtype(working_data_copy[selected_eda_feature]):
                                if imputation_method == 'Mean':
                                    working_data_copy[selected_eda_feature].fillna(
                                        working_data_copy[selected_eda_feature].mean(), inplace=True)
                                elif imputation_method == 'Median':
                                    working_data_copy[selected_eda_feature].fillna(
                                        working_data_copy[selected_eda_feature].median(), inplace=True)
                                elif imputation_method == 'Mode':
                                    mode_val = working_data_copy[selected_eda_feature].mode()
                                    if len(mode_val) > 0:
                                        working_data_copy[selected_eda_feature].fillna(mode_val[0], inplace=True)
                                elif imputation_method == 'Forward Fill':
                                    working_data_copy[selected_eda_feature].fillna(method='ffill', inplace=True)
                                elif imputation_method == 'Backward Fill':
                                    working_data_copy[selected_eda_feature].fillna(method='bfill', inplace=True)
                        
                        st.session_state.modified_historical_data = working_data_copy
                        st.success(f"Missing values in {selected_eda_feature} handled using {imputation_method}.")
                        st.rerun()
            
            # Enhanced Outlier Detection
            if pd.api.types.is_numeric_dtype(filtered_data[selected_eda_feature]):
                with st.expander("🎯 Advanced Outlier Detection", expanded=False):
                    st.markdown("### Multi-Method Outlier Detection")
                    
                    # Multiple outlier detection methods
                    outlier_methods = st.multiselect(
                        "Select outlier detection methods:",
                        ["IQR Method", "Z-Score", "Modified Z-Score", "Isolation Forest"],
                        default=["IQR Method"],
                        key='outlier_methods'
                    )
                    
                    outlier_results = {}
                    
                    for method in outlier_methods:
                        if method == "IQR Method":
                            outlier_mask, lower_bound, upper_bound = detect_outliers_iqr(filtered_data[selected_eda_feature])
                            outlier_results[method] = {
                                'mask': outlier_mask,
                                'count': outlier_mask.sum(),
                                'percentage': (outlier_mask.sum() / len(filtered_data) * 100).round(2)
                            }
                        
                        elif method == "Z-Score":
                            z_scores = np.abs(stats.zscore(filtered_data[selected_eda_feature].dropna()))
                            z_threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1, key='z_threshold')
                            outlier_mask = pd.Series(False, index=filtered_data.index)
                            outlier_mask.iloc[filtered_data[selected_eda_feature].dropna().index] = z_scores > z_threshold
                            outlier_results[method] = {
                                'mask': outlier_mask,
                                'count': outlier_mask.sum(),
                                'percentage': (outlier_mask.sum() / len(filtered_data) * 100).round(2)
                            }
                        
                        elif method == "Modified Z-Score":
                            median = filtered_data[selected_eda_feature].median()
                            mad = np.median(np.abs(filtered_data[selected_eda_feature] - median))
                            modified_z_scores = 0.6745 * (filtered_data[selected_eda_feature] - median) / mad
                            mod_z_threshold = st.slider("Modified Z-Score Threshold", 2.0, 4.0, 3.5, 0.1, key='mod_z_threshold')
                            outlier_mask = np.abs(modified_z_scores) > mod_z_threshold
                            outlier_results[method] = {
                                'mask': outlier_mask,
                                'count': outlier_mask.sum(),
                                'percentage': (outlier_mask.sum() / len(filtered_data) * 100).round(2)
                            }
                        
                        elif method == "Isolation Forest":
                            try:
                                from sklearn.ensemble import IsolationForest
                                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                                outlier_pred = iso_forest.fit_predict(filtered_data[[selected_eda_feature]].fillna(filtered_data[selected_eda_feature].mean()))
                                outlier_mask = pd.Series(outlier_pred == -1, index=filtered_data.index)
                                outlier_results[method] = {
                                    'mask': outlier_mask,
                                    'count': outlier_mask.sum(),
                                    'percentage': (outlier_mask.sum() / len(filtered_data) * 100).round(2)
                                }
                            except ImportError:
                                st.warning("scikit-learn not available for Isolation Forest")
                    
                    # Display outlier detection results
                    if outlier_results:
                        st.markdown("#### Outlier Detection Results")
                        
                        results_data = []
                        for method, result in outlier_results.items():
                            results_data.append({
                                'Method': method,
                                'Outliers Found': result['count'],
                                'Percentage': f"{result['percentage']}%"
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Visualize outliers
                        if 'date' in filtered_data.columns:
                            fig_outliers = go.Figure()
                            
                            # Add original data
                            fig_outliers.add_trace(go.Scatter(
                                x=filtered_data['date'],
                                y=filtered_data[selected_eda_feature],
                                mode='markers',
                                name='Normal',
                                marker=dict(color='blue', size=4)
                            ))
                            
                            # Add outliers for each method
                            colors = ['red', 'orange', 'purple', 'green']
                            for i, (method, result) in enumerate(outlier_results.items()):
                                outlier_data = filtered_data[result['mask']]
                                if len(outlier_data) > 0:
                                    fig_outliers.add_trace(go.Scatter(
                                        x=outlier_data['date'],
                                        y=outlier_data[selected_eda_feature],
                                        mode='markers',
                                        name=f'{method} Outliers',
                                        marker=dict(color=colors[i % len(colors)], size=8, symbol='diamond')
                                    ))
                            
                            fig_outliers.update_layout(
                                title="Outlier Detection Comparison",
                                xaxis_title="Date",
                                yaxis_title=selected_eda_feature
                            )
                            st.plotly_chart(fig_outliers, use_container_width=True)
        
        # Correlation Analysis (Enhanced)
        with st.expander("🔗 Advanced Correlation Analysis", expanded=False):
            st.markdown("### Correlation Analysis")
            
            # Feature selection for correlation
            numeric_features = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
            selected_corr_features = st.multiselect(
                "Select features for correlation analysis:",
                options=numeric_features,
                default=numeric_features[:min(5, len(numeric_features))],
                key='enhanced_corr_features'
            )
            
            if len(selected_corr_features) >= 2:
                corr_matrix = filtered_data[selected_corr_features].corr()
                
                # Correlation heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix Heatmap"
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation insights
                st.markdown("#### Correlation Insights")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j],
                            'Strength': abs(corr_matrix.iloc[i, j])
                        })
                
                corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Strength', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strongest Positive Correlations**")
                    positive_corr = corr_pairs_df[corr_pairs_df['Correlation'] > 0].head(5)
                    st.dataframe(positive_corr[['Feature 1', 'Feature 2', 'Correlation']], 
                               use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Strongest Negative Correlations**")
                    negative_corr = corr_pairs_df[corr_pairs_df['Correlation'] < 0].head(5)
                    st.dataframe(negative_corr[['Feature 1', 'Feature 2', 'Correlation']], 
                               use_container_width=True, hide_index=True)
        
        # Comparative Analysis
        if analysis_mode == "Comparative Analysis":
            with st.expander("📊 Comparative Analysis", expanded=True):
                st.markdown("### Multi-Feature Comparison")
                
                # Select features for comparison
                comparison_features = st.multiselect(
                    "Select features to compare:",
                    options=eda_feature_options,
                    default=eda_feature_options[:min(3, len(eda_feature_options))],
                    key='comparison_features'
                )
                
                if len(comparison_features) >= 2:
                    # Normalized comparison
                    scaler = StandardScaler()
                    scaled_data = pd.DataFrame(
                        scaler.fit_transform(filtered_data[comparison_features]),
                        columns=comparison_features,
                        index=filtered_data.index
                    )
                    
                    if 'date' in filtered_data.columns:
                        scaled_data['date'] = filtered_data['date']
                        
                        # Multi-line plot
                        fig_comparison = go.Figure()
                        for feature in comparison_features:
                            fig_comparison.add_trace(go.Scatter(
                                x=scaled_data['date'],
                                y=scaled_data[feature],
                                name=feature,
                                mode='lines'
                            ))
                        
                        fig_comparison.update_layout(
                            title="Normalized Feature Comparison Over Time",
                            xaxis_title="Date",
                            yaxis_title="Normalized Values"
                        )
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Statistical comparison
                    st.markdown("#### Statistical Comparison")
                    comparison_stats = filtered_data[comparison_features].describe().T
                    st.dataframe(comparison_stats, use_container_width=True)
        
        # Data Reset Option
        st.divider()
        st.markdown("## ⚙️ Data Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset to Original Data", help="Reset all modifications and return to original dataset"):
                st.session_state.modified_historical_data = historical_data.copy()
                st.success("Data reset to original state.")
                st.rerun()
        
        with col2:
            if st.button("📥 Export Current Data", help="Download modified dataset"):
                csv = st.session_state.modified_historical_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="modified_data.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📊 Generate Report", help="Generate comprehensive EDA report"):
                st.info("EDA Report generation feature coming soon!") 