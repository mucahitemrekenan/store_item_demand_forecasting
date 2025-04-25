"""
Central configuration file for the forecasting project.
"""
import os

# --- Data Configuration ---
DATA_DIR = 'data' # Example path, adjust if needed
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
SAMPLE_SUBMISSION_FILENAME = 'sample_submission.csv'
STORE_FILENAME = 'stores.csv' # Assuming store info might be relevant later
ITEM_FILENAME = 'items.csv' # Assuming item info might be relevant later

# Raw Data Columns to check
RAW_TRAIN_COLS = ['date', 'store', 'item', 'sales']
RAW_TEST_COLS = ['date', 'store', 'item', 'id']

# --- Feature Engineering Configuration ---
TARGET_COL = 'sales'
GROUP_COLS = ['store', 'item'] # Columns to group by for lags/rolling features
DATE_COL = 'date'

# Date Features
CREATE_DATE_FEATURES = True # Toggle creation of all date features
INCLUDE_CYCLICAL_DATE_FEATURES = True # sin/cos for month, dayofweek
ADD_HOLIDAY_FEATURES = True # Requires 'holidays' package

# Lag Features
CREATE_LAG_FEATURES = True
# Specify lags in days
LAG_FEATURES_LIST = [91, 98, 105, 112, 119, 126, 182, 364, 546, 728] # Example lags (match modeling.py)
LAG_FILL_METHOD = None # e.g., 'ffill' or None

# Rolling Window Features
CREATE_ROLLING_FEATURES = True
ROLLING_WINDOWS = [91, 182, 365, 546] # Example windows (match modeling.py)
ROLLING_AGGS = ['mean', 'std', 'min', 'max'] # Use 'median' if appropriate and computationally feasible
ROLLING_FILL_METHOD = None # e.g., 'ffill' or None

# Exponentially Weighted Moving Average (EWM) Features
CREATE_EWM_FEATURES = False # Set to True if you want EWM features
EWM_SPANS = [7, 14, 28, 91] # Example spans

# Interaction Features
CREATE_INTERACTION_FEATURES = False # Set to True if you want interaction features

# Features used for modeling (auto-generated if None)
# If None, the modeling script will typically combine all generated features.
# If specified, ensure this list matches the output of the feature engineering steps.
MODEL_FEATURES = None 

# Explicitly list features considered categorical by the model
# Ensure these are generated during feature engineering if used
CATEGORICAL_FEATURES = ['store', 'item', 'month', 'dayofweek', 'quarter', 'is_holiday', 'year'] # Example, adjust as needed


# --- Modeling Configuration ---
MODEL_TYPE = 'lightgbm' # Identifier for the model type

# LightGBM Specific Parameters
LGB_PARAMS = {
    'objective': 'regression_l1', # MAE
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

# Cross-Validation Settings
USE_TIMESERIES_CV = True
CV_FOLDS = 5 # Number of folds for TimeSeriesSplit
# Optional: Define gap between train and validation in TimeSeriesSplit
# CV_GAP = 0 

# Model Training Settings
RETRAIN_ON_FULL_DATA = True # Retrain final model on all available data after CV
EARLY_STOPPING_ROUNDS = 50 # For LightGBM callbacks

# --- Output Configuration ---
MODEL_OUTPUT_DIR = 'models'
# Filename template for saving models (can include placeholders like {store}, {item}, {timestamp})
MODEL_FILENAME_TEMPLATE = 'lgbm_model_{timestamp}.joblib' 
# Example: 'lgbm_model_store{store}_item{item}.joblib' if training per group
# Example: 'lgbm_global_model_{timestamp}.joblib' for a single global model

# Create output directory if it doesn't exist
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

# --- Forecasting Configuration ---
FORECAST_HORIZON = 90 # Days to forecast into the future (e.g., for Streamlit app)

# --- Streamlit App Configuration ---
# Define any specific configurations for the Streamlit app here
APP_TITLE = "Store Item Demand Forecasting" 