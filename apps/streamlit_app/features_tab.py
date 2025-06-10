# Contents for Features tab - Enhanced Feature Engineering Interface
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def create_time_features(data, date_col='date'):
    """Create comprehensive time-based features."""
    if date_col not in data.columns:
        return data
    
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Cyclical features (encoded as sine/cosine)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Business calendar features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    # Days since reference
    reference_date = df[date_col].min()
    df['days_since_start'] = (df[date_col] - reference_date).dt.days
    
    return df


def create_lag_features(data, target_col, lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for a target column."""
    df = data.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(data, target_col, windows=[3, 7, 14, 30], stats=['mean', 'std', 'min', 'max']):
    """Create rolling window statistical features."""
    df = data.copy()
    
    for window in windows:
        for stat in stats:
            if stat == 'mean':
                df[f'{target_col}_rolling_{window}_mean'] = df[target_col].rolling(window=window).mean()
            elif stat == 'std':
                df[f'{target_col}_rolling_{window}_std'] = df[target_col].rolling(window=window).std()
            elif stat == 'min':
                df[f'{target_col}_rolling_{window}_min'] = df[target_col].rolling(window=window).min()
            elif stat == 'max':
                df[f'{target_col}_rolling_{window}_max'] = df[target_col].rolling(window=window).max()
            elif stat == 'median':
                df[f'{target_col}_rolling_{window}_median'] = df[target_col].rolling(window=window).median()
            elif stat == 'skew':
                df[f'{target_col}_rolling_{window}_skew'] = df[target_col].rolling(window=window).skew()
    
    return df


def create_difference_features(data, target_col, periods=[1, 7, 30]):
    """Create difference features."""
    df = data.copy()
    
    for period in periods:
        df[f'{target_col}_diff_{period}'] = df[target_col].diff(periods=period)
        df[f'{target_col}_pct_change_{period}'] = df[target_col].pct_change(periods=period)
    
    return df


def calculate_feature_importance(data, target_col, feature_cols):
    """Calculate feature importance using Random Forest."""
    try:
        # Prepare data
        X = data[feature_cols].fillna(data[feature_cols].mean())
        y = data[target_col].fillna(data[target_col].mean())
        
        # Remove rows where target is still NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            return pd.DataFrame()
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        st.warning(f"Could not calculate feature importance: {e}")
        return pd.DataFrame()


def render_features_tab(historical_data, lag_feature_names, rolling_feature_names):
    """Renders the enhanced Features tab content."""
    
    # Initialize session state for feature-engineered data
    if 'feature_engineered_data' not in st.session_state:
        st.session_state.feature_engineered_data = historical_data.copy()
    
    # Sidebar for feature engineering controls
    with st.sidebar:
        st.markdown("### ⚙️ Feature Engineering Controls")
        
        # Target variable selection
        numeric_cols = historical_data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_variable = st.selectbox(
                "Target Variable:",
                options=numeric_cols,
                index=numeric_cols.index('sales') if 'sales' in numeric_cols else 0,
                key='target_variable'
            )
        else:
            target_variable = None
        
        # Feature categories
        st.markdown("#### Feature Categories")
        create_time_feats = st.checkbox("Time-based Features", value=True, key='create_time_feats')
        create_lag_feats = st.checkbox("Lag Features", value=True, key='create_lag_feats')
        create_rolling_feats = st.checkbox("Rolling Features", value=True, key='create_rolling_feats')
        create_diff_feats = st.checkbox("Difference Features", value=False, key='create_diff_feats')
    
    with st.container():
        st.title("🔧 Feature Engineering")
        st.markdown("**Advanced feature creation and analysis dashboard**")
        
        if historical_data.empty:
            st.warning("No historical data to generate features from.")
            return
        
        # Feature Engineering Pipeline
        st.markdown("## 🏗️ Feature Engineering Pipeline")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Current Dataset")
            st.dataframe(historical_data.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### Dataset Info")
            st.metric("Total Records", len(historical_data))
            st.metric("Current Features", len(historical_data.columns))
            if 'date' in historical_data.columns:
                date_range = (historical_data['date'].max() - historical_data['date'].min()).days
                st.metric("Date Range (days)", date_range)
        
        st.divider()
        
        # Feature Creation Section
        with st.expander("🏭 Feature Creation Workshop", expanded=True):
            
            # Time-based Features
            if create_time_feats and 'date' in historical_data.columns:
                st.markdown("#### ⏰ Time-based Features")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    include_cyclical = st.checkbox("Cyclical Features (sin/cos)", value=True, key='cyclical')
                with col2:
                    include_business = st.checkbox("Business Calendar", value=True, key='business')
                with col3:
                    if st.button("Create Time Features", key='create_time_btn'):
                        temp_data = create_time_features(st.session_state.feature_engineered_data)
                        st.session_state.feature_engineered_data = temp_data
                        st.success("Time-based features created!")
                        st.rerun()
            
            # Lag Features
            if create_lag_feats and target_variable:
                st.markdown("#### 📈 Lag Features")
                col1, col2 = st.columns(2)
                
                with col1:
                    lag_periods = st.multiselect(
                        "Lag Periods:",
                        options=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 60, 90],
                        default=[1, 2, 3, 7, 14, 30],
                        key='lag_periods'
                    )
                
                with col2:
                    if st.button("Create Lag Features", key='create_lag_btn'):
                        if lag_periods:
                            temp_data = create_lag_features(
                                st.session_state.feature_engineered_data, 
                                target_variable, 
                                lag_periods
                            )
                            st.session_state.feature_engineered_data = temp_data
                            st.success(f"Lag features created for periods: {lag_periods}")
                            st.rerun()
            
            # Rolling Features
            if create_rolling_feats and target_variable:
                st.markdown("#### 📊 Rolling Window Features")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rolling_windows = st.multiselect(
                        "Window Sizes:",
                        options=[3, 5, 7, 10, 14, 21, 30, 60, 90],
                        default=[3, 7, 14, 30],
                        key='rolling_windows'
                    )
                
                with col2:
                    rolling_stats = st.multiselect(
                        "Statistics:",
                        options=['mean', 'std', 'min', 'max', 'median', 'skew'],
                        default=['mean', 'std', 'min', 'max'],
                        key='rolling_stats'
                    )
                
                with col3:
                    if st.button("Create Rolling Features", key='create_rolling_btn'):
                        if rolling_windows and rolling_stats:
                            temp_data = create_rolling_features(
                                st.session_state.feature_engineered_data,
                                target_variable,
                                rolling_windows,
                                rolling_stats
                            )
                            st.session_state.feature_engineered_data = temp_data
                            st.success(f"Rolling features created!")
                            st.rerun()
            
            # Difference Features
            if create_diff_feats and target_variable:
                st.markdown("#### 📐 Difference Features")
                col1, col2 = st.columns(2)
                
                with col1:
                    diff_periods = st.multiselect(
                        "Difference Periods:",
                        options=[1, 2, 3, 7, 14, 30],
                        default=[1, 7],
                        key='diff_periods'
                    )
                
                with col2:
                    if st.button("Create Difference Features", key='create_diff_btn'):
                        if diff_periods:
                            temp_data = create_difference_features(
                                st.session_state.feature_engineered_data,
                                target_variable,
                                diff_periods
                            )
                            st.session_state.feature_engineered_data = temp_data
                            st.success("Difference features created!")
                            st.rerun()
        
        st.divider()
        
        # Feature Analysis Section
        st.markdown("## 📊 Feature Analysis & Selection")
        
        # Current feature summary
        current_features = st.session_state.feature_engineered_data.columns.tolist()
        numeric_features = st.session_state.feature_engineered_data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(current_features))
        with col2:
            st.metric("Numeric Features", len(numeric_features))
        with col3:
            new_features = len(current_features) - len(historical_data.columns)
            st.metric("New Features", new_features)
        with col4:
            missing_pct = (st.session_state.feature_engineered_data.isnull().sum().sum() / 
                          (len(st.session_state.feature_engineered_data) * len(current_features)) * 100)
            st.metric("Missing %", f"{missing_pct:.2f}%")
        
        # Feature Importance Analysis
        with st.expander("📈 Feature Importance Analysis", expanded=False):
            if target_variable and len(numeric_features) > 1:
                feature_cols_for_importance = [col for col in numeric_features if col != target_variable]
                
                if len(feature_cols_for_importance) > 0:
                    importance_df = calculate_feature_importance(
                        st.session_state.feature_engineered_data,
                        target_variable,
                        feature_cols_for_importance
                    )
                    
                    if not importance_df.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Top 15 Most Important Features")
                            top_features = importance_df.head(15)
                            st.dataframe(top_features, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.markdown("#### Feature Importance Plot")
                            fig_importance = px.bar(
                                top_features.head(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 10 Feature Importance"
                            )
                            fig_importance.update_layout(height=400)
                            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feature Visualization
        with st.expander("📊 Feature Visualization", expanded=False):
            if len(numeric_features) > 1:
                viz_type = st.selectbox(
                    "Visualization Type:",
                    ["Feature Distribution", "Feature Correlation", "Feature vs Target", "Feature Timeline"],
                    key='viz_type'
                )
                
                if viz_type == "Feature Distribution":
                    selected_feature = st.selectbox(
                        "Select Feature:",
                        options=numeric_features,
                        key='feature_for_dist'
                    )
                    
                    if selected_feature:
                        fig_dist = px.histogram(
                            st.session_state.feature_engineered_data,
                            x=selected_feature,
                            title=f"Distribution of {selected_feature}",
                            marginal="box"
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                elif viz_type == "Feature Correlation":
                    correlation_features = st.multiselect(
                        "Select Features for Correlation:",
                        options=numeric_features,
                        default=numeric_features[:min(5, len(numeric_features))],
                        key='corr_features'
                    )
                    
                    if len(correlation_features) >= 2:
                        corr_matrix = st.session_state.feature_engineered_data[correlation_features].corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            title="Feature Correlation Matrix"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                
                elif viz_type == "Feature vs Target" and target_variable:
                    feature_for_scatter = st.selectbox(
                        "Select Feature:",
                        options=[col for col in numeric_features if col != target_variable],
                        key='feature_for_scatter'
                    )
                    
                    if feature_for_scatter:
                        fig_scatter = px.scatter(
                            st.session_state.feature_engineered_data,
                            x=feature_for_scatter,
                            y=target_variable,
                            title=f"{feature_for_scatter} vs {target_variable}",
                            trendline="ols"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                elif viz_type == "Feature Timeline" and 'date' in st.session_state.feature_engineered_data.columns:
                    timeline_features = st.multiselect(
                        "Select Features for Timeline:",
                        options=numeric_features,
                        default=[numeric_features[0]] if numeric_features else [],
                        key='timeline_features'
                    )
                    
                    if timeline_features:
                        fig_timeline = go.Figure()
                        for feature in timeline_features:
                            fig_timeline.add_trace(go.Scatter(
                                x=st.session_state.feature_engineered_data['date'],
                                y=st.session_state.feature_engineered_data[feature],
                                name=feature,
                                mode='lines'
                            ))
                        
                        fig_timeline.update_layout(
                            title="Feature Timeline",
                            xaxis_title="Date",
                            yaxis_title="Value"
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Dataset Preview and Export
        st.divider()
        st.markdown("## 📋 Feature-Engineered Dataset")
        
        # Dataset preview with feature highlighting
        st.markdown("### Dataset Preview")
        preview_cols = st.multiselect(
            "Select columns to preview:",
            options=current_features,
            default=current_features[:min(8, len(current_features))],
            key='preview_cols'
        )
        
        if preview_cols:
            st.dataframe(
                st.session_state.feature_engineered_data[preview_cols].head(20),
                use_container_width=True
            )
        
        # Data Management
        st.markdown("### 🛠️ Data Management")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Reset to Original", key='reset_features'):
                st.session_state.feature_engineered_data = historical_data.copy()
                st.success("Reset to original dataset!")
                st.rerun()
        
        with col2:
            if st.button("📥 Export Features", key='export_features'):
                csv = st.session_state.feature_engineered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="feature_engineered_data.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🧹 Clean Missing Values", key='clean_missing'):
                # Simple cleaning: forward fill then backward fill
                temp_data = st.session_state.feature_engineered_data.fillna(method='ffill').fillna(method='bfill')
                st.session_state.feature_engineered_data = temp_data
                st.success("Missing values cleaned!")
                st.rerun()
        
        with col4:
            if st.button("📊 Feature Report", key='feature_report'):
                st.info("Feature engineering report generation coming soon!")
        
        # Feature Engineering Summary
        with st.expander("📈 Feature Engineering Summary", expanded=False):
            if len(current_features) > len(historical_data.columns):
                st.markdown("#### Features Added")
                original_features = set(historical_data.columns)
                new_features = [col for col in current_features if col not in original_features]
                
                if new_features:
                    # Categorize new features
                    time_features = [f for f in new_features if any(keyword in f for keyword in ['year', 'month', 'day', 'week', 'sin', 'cos', 'is_'])]
                    lag_features = [f for f in new_features if 'lag' in f]
                    rolling_features = [f for f in new_features if 'rolling' in f]
                    diff_features = [f for f in new_features if any(keyword in f for keyword in ['diff', 'pct_change'])]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if time_features:
                            st.markdown("**Time Features:**")
                            st.write(f"• {len(time_features)} features")
                            if st.checkbox("Show time features", key='show_time_features'):
                                for f in time_features:
                                    st.text(f"  - {f}")
                    
                    with col2:
                        if lag_features:
                            st.markdown("**Lag Features:**")
                            st.write(f"• {len(lag_features)} features")
                            if st.checkbox("Show lag features", key='show_lag_features'):
                                for f in lag_features:
                                    st.text(f"  - {f}")
                    
                    with col3:
                        if rolling_features:
                            st.markdown("**Rolling Features:**")
                            st.write(f"• {len(rolling_features)} features")
                            if st.checkbox("Show rolling features", key='show_rolling_features'):
                                for f in rolling_features:
                                    st.text(f"  - {f}")
                        
                        if diff_features:
                            st.markdown("**Difference Features:**")
                            st.write(f"• {len(diff_features)} features")
                            if st.checkbox("Show difference features", key='show_diff_features'):
                                for f in diff_features:
                                    st.text(f"  - {f}") 