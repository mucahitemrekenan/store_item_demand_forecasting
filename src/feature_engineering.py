"""
Module for creating time series features.
"""

import pandas as pd
import numpy as np
import logging

# Import configurations
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_date_features(df: pd.DataFrame, date_col: str = 'date', include_cyclical: bool = True, add_holidays: bool = True) -> pd.DataFrame:
    """Creates date-based features from a date column.

    Args:
        df: DataFrame containing the date column.
        date_col: The name of the date column.
        include_cyclical: Whether to include cyclical features.
        add_holidays: Whether to add holiday flags.

    Returns:
        DataFrame with added date features.
    """
    if date_col not in df.columns:
        logging.error(f"Date column '{date_col}' not found in DataFrame.")
        return df # Return original df if date column is missing
    
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Ensure the date column is in datetime format
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    except Exception as e:
        logging.error(f"Could not convert column '{date_col}' to datetime: {e}")
        return df # Return original df on error

    logging.info(f"Creating date features from column '{date_col}' (cyclical={include_cyclical}, holidays={add_holidays})...")
    df_copy['year'] = df_copy[date_col].dt.year
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['day'] = df_copy[date_col].dt.day
    df_copy['dayofweek'] = df_copy[date_col].dt.dayofweek # Monday=0, Sunday=6
    df_copy['dayofyear'] = df_copy[date_col].dt.dayofyear
    df_copy['weekofyear'] = df_copy[date_col].dt.isocalendar().week.astype(int)
    df_copy['quarter'] = df_copy[date_col].dt.quarter
    # Month/Quarter end flags
    df_copy['is_month_end'] = df_copy[date_col].dt.is_month_end.astype(int)
    df_copy['is_quarter_end'] = df_copy[date_col].dt.is_quarter_end.astype(int)
    # Cyclical features
    if include_cyclical:
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['dayofweek'] / 7)
        df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['dayofweek'] / 7)
    # Holiday flags
    if add_holidays:
        try:
            import holidays
            us_holidays = holidays.CountryHoliday('US')
            df_copy['is_holiday'] = df_copy[date_col].dt.date.apply(lambda d: int(d in us_holidays))
        except ImportError:
            logging.warning("holidays package not installed, skipping holiday flags.")
            df_copy['is_holiday'] = 0

    logging.info("Date features created successfully.")
    return df_copy

def create_lag_features(df: pd.DataFrame, group_cols: list[str], target_col: str, lags: list[int], fill_method: str = None) -> pd.DataFrame:
    """Creates lag features for a target column, grouped by specified columns.

    Args:
        df: DataFrame.
        group_cols: List of columns to group by (e.g., ['store', 'item']).
        target_col: The column to create lag features for (e.g., 'sales').
        lags: A list of lag periods (e.g., [1, 7, 14]).
        fill_method: Method to fill NaNs (e.g., 'ffill', 'bfill').

    Returns:
        DataFrame with added lag features.
    """
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found.")
        return df
    if not all(col in df.columns for col in group_cols):
        logging.error(f"One or more group columns ({group_cols}) not found.")
        return df
    
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    logging.info(f"Creating lag features for '{target_col}' grouped by {group_cols} for lags: {lags}")
    
    # Sort by date within groups to ensure correct lag calculation
    # Assumes date is either the index or a column
    date_col_name = df_copy.index.name if df_copy.index.name else 'date'
    if date_col_name not in df_copy.columns and df_copy.index.name != date_col_name:
         logging.warning(f"Date column/index '{date_col_name}' not found for sorting before lag calculation.")
         # Attempt to sort anyway if index is datetime
         if isinstance(df_copy.index, pd.DatetimeIndex):
             df_copy = df_copy.sort_index()
         else:
            logging.error("Cannot determine date column/index for sorting. Lags might be incorrect.")
            # We might still proceed, but with a strong warning
    else:
        # Sort by group keys and date
        sort_cols = group_cols + [date_col_name]
        df_copy = df_copy.sort_values(sort_cols)

    for lag in lags:
        lag_col_name = f'{target_col}_lag_{lag}'
        series = df_copy.groupby(group_cols)[target_col].shift(lag)
        if fill_method:
            df_copy[lag_col_name] = series.fillna(method=fill_method)
        else:
            df_copy[lag_col_name] = series

    logging.info("Lag features created.")
    return df_copy

# --- Add functions for rolling window features, exponential smoothing features, etc. below ---

def create_rolling_features(df: pd.DataFrame, group_cols: list[str], target_col: str, windows: list[int], aggs: list[str], fill_method: str = None) -> pd.DataFrame:
    """Creates rolling window features with optional NaN fill."""
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found.")
        return df
    if not all(col in df.columns for col in group_cols):
        logging.error(f"One or more group columns ({group_cols}) not found.")
        return df

    df_copy = df.copy()
    logging.info(f"Creating rolling features for '{target_col}' grouped by {group_cols}...")

    # Assumes sorting is done, or done in lag features if called prior
    grouped = df_copy.groupby(group_cols)[target_col]

    for window in windows:
        # Shift by 1 to prevent data leakage (use past data only)
        shifted = grouped.shift(1)
        rolling = shifted.rolling(window=window, min_periods=max(1, window // 2)) # Adjust min_periods as needed
        
        for agg in aggs:
            try:
                col_name = f'{target_col}_roll_{agg}_{window}d'
                df_copy[col_name] = getattr(rolling, agg)()
                if fill_method:
                    df_copy[col_name] = df_copy[col_name].fillna(method=fill_method)
            except Exception as e:
                logging.warning(f"Could not compute rolling {agg} for window {window}: {e}")

    logging.info("Rolling features created.")
    return df_copy

def create_expanding_features(df: pd.DataFrame, group_cols: list[str], target_col: str, aggs: list[str]) -> pd.DataFrame:
    """Creates expanding window features."""
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found.")
        return df
    if not all(col in df.columns for col in group_cols):
        logging.error(f"One or more group columns ({group_cols}) not found.")
        return df

    df_copy = df.copy()
    logging.info(f"Creating expanding features for '{target_col}' grouped by {group_cols}...")

    grouped = df_copy.groupby(group_cols)[target_col]

    for agg in aggs:
        col = f'{target_col}_expanding_{agg}'
        df_copy[col] = getattr(grouped.expanding(min_periods=1), agg)().reset_index(level=group_cols, drop=True)

    logging.info("Expanding features created.")
    return df_copy

def create_ewm_features(df: pd.DataFrame, group_cols: list[str], target_col: str, spans: list[int]) -> pd.DataFrame:
    """Creates exponentially weighted moving average features."""
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found.")
        return df
    if not all(col in df.columns for col in group_cols):
        logging.error(f"One or more group columns ({group_cols}) not found.")
        return df

    df_copy = df.copy()
    logging.info(f"Creating EWM features for '{target_col}' grouped by {group_cols}...")

    for span in spans:
        col = f'{target_col}_ewm_{span}'
        df_copy[col] = df_copy.groupby(group_cols)[target_col].transform(lambda s: s.ewm(span=span, adjust=False).mean())

    logging.info("EWM features created.")
    return df_copy

def create_interaction_features(df: pd.DataFrame, base_cols: list[str], target_col: str, lags: list[int]) -> pd.DataFrame:
    """Creates ratio and difference interaction features between target and its lagged versions."""
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found.")
        return df
    if not all(col in df.columns for col in base_cols):
        logging.error(f"One or more base columns ({base_cols}) not found.")
        return df

    df_copy = df.copy()
    logging.info(f"Creating interaction features for '{target_col}' with base columns {base_cols} and lags {lags}...")

    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}'
        df_copy[f'{target_col}_diff_{lag}'] = df_copy[target_col] - df_copy.get(lag_col)
        df_copy[f'{target_col}_ratio_{lag}'] = df_copy[target_col] / (df_copy.get(lag_col).replace(0, np.nan))

    logging.info("Interaction features created.")
    return df_copy

# Example usage (optional)
if __name__ == '__main__':
    logging.info("Running feature_engineering module directly...")
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'date': dates.repeat(2), # Two series (e.g., store/item)
        'group': ['A'] * 100 + ['B'] * 100,
        'value': np.random.rand(200) * 100 + np.tile(np.sin(np.arange(100)/10)*10, 2)
    }
    sample_df = pd.DataFrame(data)
    sample_df['value'] = sample_df['value'].astype(int)

    print("--- Original Sample Data ---")
    print(sample_df.head())

    # Test date features
    df_with_dates = create_date_features(sample_df, date_col='date')
    print("\n--- With Date Features ---")
    print(df_with_dates.head())

    # Test lag features
    # Set date index for easier lag/rolling calculation if needed
    # sample_df_indexed = sample_df.set_index('date')
    lags_to_create = [1, 3, 7]
    df_with_lags = create_lag_features(sample_df, group_cols=['group'], target_col='value', lags=lags_to_create)
    print("\n--- With Lag Features ---")
    # Display rows where lags are likely populated
    print(df_with_lags[df_with_lags['group'] == 'A'].head(10))
    print(df_with_lags[df_with_lags['group'] == 'B'].head(10))

    # Test rolling features
    windows_to_create = [7, 14]
    aggs_to_create = ['mean', 'std']
    df_with_rolling = create_rolling_features(df_with_lags, group_cols=['group'], target_col='value', windows=windows_to_create, aggs=aggs_to_create)
    print("\n--- With Rolling Features ---")
    print(df_with_rolling[df_with_rolling['group'] == 'A'].head(15))
    print(df_with_rolling.info())

def create_features_for_prediction(historical_data: pd.DataFrame, future_idx: pd.DatetimeIndex, store_id: int, item_id: int) -> pd.DataFrame | None:
    """
    Generates features for future dates based on historical data.

    This function creates a future DataFrame, combines it with historical data, 
    applies various feature engineering steps (based on config.py settings), 
    and returns the features for the future dates only.

    Args:
        historical_data: DataFrame with historical sales data (must contain 'date', 'store', 'item', 'sales').
        future_idx: DatetimeIndex for the future dates to predict.
        store_id: The specific store ID for the forecast.
        item_id: The specific item ID for the forecast.

    Returns:
        DataFrame containing features for the future dates, or None if an error occurs.
    """
    logging.info(f"Generating features for future dates for store {store_id}, item {item_id}...")

    required_hist_cols = [config.DATE_COL] + config.GROUP_COLS + [config.TARGET_COL]
    if historical_data.empty or not all(col in historical_data.columns for col in required_hist_cols):
        logging.error(f"Historical data is empty or missing required columns ({required_hist_cols}).")
        return None
    
    # Ensure historical data date column is datetime
    try:
        historical_data[config.DATE_COL] = pd.to_datetime(historical_data[config.DATE_COL])
    except Exception as e:
        logging.error(f"Could not convert historical data '{config.DATE_COL}' column to datetime: {e}")
        return None

    # Create a DataFrame for the future dates
    future_df = pd.DataFrame(index=future_idx)
    future_df[config.DATE_COL] = future_idx
    future_df[config.GROUP_COLS[0]] = store_id # Assuming first group col is store
    future_df[config.GROUP_COLS[1]] = item_id  # Assuming second group col is item
    future_df[config.TARGET_COL] = np.nan # Sales are unknown for the future

    # Combine historical data (relevant portion for lags/rolling) and future data
    cols_to_keep = required_hist_cols
    # Ensure consistent dtypes before concat if necessary (especially for categorical)
    # It's safer to apply categorical conversion *after* concat and feature creation
    combined_df = pd.concat([historical_data[cols_to_keep], future_df[cols_to_keep]], ignore_index=True)
    combined_df = combined_df.sort_values(config.GROUP_COLS + [config.DATE_COL]).reset_index(drop=True)

    # Apply feature engineering steps based on config
    try:
        if config.CREATE_DATE_FEATURES:
            logging.info("Applying date features...")
            combined_df = create_date_features(combined_df, date_col=config.DATE_COL,
                                            include_cyclical=config.INCLUDE_CYCLICAL_DATE_FEATURES,
                                            add_holidays=config.ADD_HOLIDAY_FEATURES)

        if config.CREATE_LAG_FEATURES:
            logging.info(f"Applying lag features: {config.LAG_FEATURES_LIST}")
            combined_df = create_lag_features(combined_df, group_cols=config.GROUP_COLS, 
                                            target_col=config.TARGET_COL, lags=config.LAG_FEATURES_LIST, 
                                            fill_method=config.LAG_FILL_METHOD)

        if config.CREATE_ROLLING_FEATURES:
            logging.info(f"Applying rolling features: windows={config.ROLLING_WINDOWS}, aggs={config.ROLLING_AGGS}")
            combined_df = create_rolling_features(combined_df, group_cols=config.GROUP_COLS, 
                                                target_col=config.TARGET_COL, windows=config.ROLLING_WINDOWS, 
                                                aggs=config.ROLLING_AGGS, fill_method=config.ROLLING_FILL_METHOD)
        
        if config.CREATE_EWM_FEATURES:
            logging.info(f"Applying EWM features: spans={config.EWM_SPANS}")
            combined_df = create_ewm_features(combined_df, group_cols=config.GROUP_COLS, 
                                            target_col=config.TARGET_COL, spans=config.EWM_SPANS)

        if config.CREATE_INTERACTION_FEATURES:
            logging.info("Applying interaction features...")
            combined_df = create_interaction_features(combined_df, base_cols=config.GROUP_COLS, 
                                                    target_col=config.TARGET_COL, lags=config.LAG_FEATURES_LIST) # Assuming interactions use same lags

    except Exception as e:
        logging.error(f"Error during feature engineering for prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Filter out only the future dates
    features_for_future = combined_df[combined_df[config.DATE_COL].isin(future_idx)].copy()
    
    # Convert categorical features *after* filtering for future dates
    # This ensures consistency if categories appear only in history/future
    for col in config.CATEGORICAL_FEATURES:
         if col in features_for_future.columns:
            # Convert to pandas Categorical; consistency with training is handled by model usually
            # Or load categories from training if needed
            features_for_future[col] = pd.Categorical(features_for_future[col])
            # If LightGBM needs specific category codes, mapping might be required here
            # based on categories saved during training.

    logging.info(f"Successfully generated {features_for_future.shape[0]} rows of features for prediction.")
    
    # Note: The calling function (forecast_tab.py) is responsible for selecting 
    # the exact feature columns the loaded model expects (using model.feature_name_).
    # We return all generated features here.

    return features_for_future 