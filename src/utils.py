# apps/streamlit_app/utils.py
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import joblib
import os
import logging
from sklearn.ensemble import RandomForestRegressor


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


def detect_outliers_iqr(data: pd.Series) -> tuple[pd.Series, float, float]:
    """Detect outliers using IQR method.
    
    Returns:
        Tuple of (boolean mask of outliers, lower bound, upper bound)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return outlier_mask, lower_bound, upper_bound


def perform_stationarity_tests(data):
    """Perform stationarity tests on time series data."""
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return None, None
        
        # ADF Test
        adf_result = adfuller(clean_data)
        adf_stats = {
            'test_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        # KPSS Test
        kpss_result = kpss(clean_data, regression='c')
        kpss_stats = {
            'test_statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
        
        return adf_stats, kpss_stats
    except ImportError:
        st.warning("statsmodels not installed. Stationarity tests unavailable.")
        return None, None
    except Exception as e:
        st.warning(f"Error in stationarity tests: {e}")
        return None, None


def perform_seasonality_decomposition(data, date_col):
    """Perform seasonal decomposition of time series."""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare data
        ts_data = data.set_index(date_col).asfreq('D')
        ts_data = ts_data.interpolate()
        
        if len(ts_data) < 14:
            return None
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
        
        return {
            'original': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    except ImportError:
        st.warning("statsmodels not installed. Seasonality decomposition unavailable.")
        return None
    except Exception as e:
        st.warning(f"Error in seasonality decomposition: {e}")
        return None


def calculate_advanced_statistics(data):
    """Calculate advanced statistical measures."""
    stats_dict = {}
    
    # Basic statistics
    stats_dict['mean'] = data.mean()
    stats_dict['median'] = data.median()
    stats_dict['std'] = data.std()
    stats_dict['variance'] = data.var()
    stats_dict['skewness'] = stats.skew(data.dropna())
    stats_dict['kurtosis'] = stats.kurtosis(data.dropna())
    
    # Distribution tests
    try:
        shapiro_stat, shapiro_p = stats.shapiro(data.dropna()[:5000])  # Limit for large datasets
        stats_dict['shapiro_stat'] = shapiro_stat
        stats_dict['shapiro_p'] = shapiro_p
        stats_dict['is_normal'] = shapiro_p > 0.05
    except:
        stats_dict['shapiro_stat'] = np.nan
        stats_dict['shapiro_p'] = np.nan
        stats_dict['is_normal'] = False
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        stats_dict[f'p{p}'] = np.percentile(data.dropna(), p)
    
    return stats_dict


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