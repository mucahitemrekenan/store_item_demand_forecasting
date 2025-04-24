"""
Module for training and evaluating forecasting models.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import logging
import os
from typing import List, Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime # Import datetime

# Import from sibling modules
from src.data_processing import load_data, prepare_data
from src.feature_engineering import create_date_features, create_lag_features, create_rolling_features
# Import configurations
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (Now Imported from config.py) ---
# Features to use (adjust based on feature engineering)
# Note: Ensure these match the columns created in feature_engineering
# DATE_FEATURES = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter'] # Defined in config
# Example lags and rolling windows - adjust as needed
# LAG_FEATURES_SPEC = { # Defined in config
#     'lags': [91, 98, 105, 112, 119, 126, 182, 364, 546, 728], # Example lags (e.g., quarterly, yearly)
# }
# ROLLING_FEATURES_SPEC = { # Defined in config
#     'windows': [91, 182, 365, 546], # Example windows (quarterly, half-yearly, etc.)
#     'aggs': ['mean', 'std', 'median']
# }

# CATEGORICAL_FEATURES = ['store', 'item', 'month', 'dayofweek', 'quarter'] # Defined in config
# TARGET_COL = 'sales' # Defined in config
# GROUP_COLS = ['store', 'item'] # Defined in config

# LGB_PARAMS = { # Defined in config
#     'objective': 'regression_l1', # MAE
#     'metric': 'mae',
#     'n_estimators': 1000,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'num_leaves': 31,
#     'verbose': -1,
#     'n_jobs': -1,
#     'seed': 42,
#     'boosting_type': 'gbdt',
# }

# CV_FOLDS = 5 # Defined in config
# MODEL_OUTPUT_DIR = 'models' # Defined in config
# MODEL_FILENAME = 'lgbm_model.joblib' # Defined in config (as template)

# --- Evaluation Metric ---
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle division by zero where both true and pred are zero
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    return np.mean(ratio) * 100

# --- Model Training ---
def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
               X_val: pd.DataFrame, y_val: pd.Series,
               features: List[str],
               params: Dict[str, Any] = config.LGB_PARAMS, # Use config
               fit_params: Dict[str, Any] = None) -> lgb.LGBMRegressor:
    """Trains a LightGBM model. Assumes categorical features are already dtype 'category'.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        features: List of feature names to use.
        params: LightGBM parameters.
        fit_params: Additional parameters for the fit method (e.g., callbacks).

    Returns:
        Trained LightGBM model.
    """
    logging.info(f"Training LightGBM model with {len(features)} features.")
    model = lgb.LGBMRegressor(**params)

    default_fit_params = {
        'eval_set': [(X_train[features], y_train), (X_val[features], y_val)],
        'eval_metric': 'mae', # Keep metric consistent with objective
        # Use callbacks for early stopping
        'callbacks': [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)], # Use config
    }
    if fit_params:
        default_fit_params.update(fit_params)

    # Removed category conversion from here - it's done before the CV loop now

    # Pass categorical_feature='auto' or the explicit list if needed,
    # but the dtype conversion is the primary fix
    model.fit(X_train[features], y_train,
              categorical_feature='auto', # Let LightGBM detect 'category' dtype
              **default_fit_params)
    logging.info("Model training complete.")
    return model

# --- Main Training Pipeline ---
def run_training_pipeline(data_path: str = config.RAW_DATA_DIR, # Use config
                        model_output_dir: str = config.MODEL_OUTPUT_DIR, # Use config
                        retrain_on_full_data: bool = config.RETRAIN_ON_FULL_DATA) -> Tuple[lgb.LGBMRegressor, Dict[str, float]]: # Use config
    """Runs the full training pipeline: load, feature engineer, CV train, evaluate, save.

    Args:
        data_path: Path to the data directory.
        model_output_dir: Directory to save the trained model.
        retrain_on_full_data: Whether to retrain the final model on the full dataset after CV.

    Returns:
        A tuple containing the final trained model and a dictionary of average CV scores.
    """
    # 1. Load Data
    logging.info("--- Starting Training Pipeline ---")
    # Adjust load_data call if it needs specific filenames from config
    train_df_raw, test_df_raw = load_data(data_path, train_file=config.TRAIN_FILENAME, test_file=config.TEST_FILENAME)
    if train_df_raw is None:
        logging.error("Failed to load data. Exiting pipeline.")
        return None, None

    # We only need train data for training/CV
    df = train_df_raw.copy()

    # 2. Feature Engineering
    logging.info("--- Starting Feature Engineering ---")
    if config.DATE_COL not in df.columns:
         if isinstance(df.index, pd.DatetimeIndex) and df.index.name == config.DATE_COL:
             df = df.reset_index()
         else:
             logging.error(f"'{config.DATE_COL}' column not found for feature engineering.")
             return None, None
    
    # Apply feature engineering based on config flags
    if config.CREATE_DATE_FEATURES:
        logging.info("Applying date features...")
        df = create_date_features(df, date_col=config.DATE_COL,
                                  include_cyclical=config.INCLUDE_CYCLICAL_DATE_FEATURES,
                                  add_holidays=config.ADD_HOLIDAY_FEATURES)
    
    if config.CREATE_LAG_FEATURES:
        logging.info(f"Applying lag features: {config.LAG_FEATURES_LIST}")
        df = create_lag_features(df, group_cols=config.GROUP_COLS, target_col=config.TARGET_COL, 
                                 lags=config.LAG_FEATURES_LIST, fill_method=config.LAG_FILL_METHOD)
    
    if config.CREATE_ROLLING_FEATURES:
        logging.info(f"Applying rolling features: windows={config.ROLLING_WINDOWS}, aggs={config.ROLLING_AGGS}")
        df = create_rolling_features(df, group_cols=config.GROUP_COLS, target_col=config.TARGET_COL,
                                   windows=config.ROLLING_WINDOWS, aggs=config.ROLLING_AGGS,
                                   fill_method=config.ROLLING_FILL_METHOD)

    # --- Add calls for EWM and Interaction features based on config flags ---
    if config.CREATE_EWM_FEATURES:
        # Need to import create_ewm_features if used
        from src.feature_engineering import create_ewm_features
        logging.info(f"Applying EWM features: spans={config.EWM_SPANS}")
        df = create_ewm_features(df, group_cols=config.GROUP_COLS, target_col=config.TARGET_COL, spans=config.EWM_SPANS)
        
    if config.CREATE_INTERACTION_FEATURES:
        # Need to import create_interaction_features if used
        from src.feature_engineering import create_interaction_features
        logging.info("Applying interaction features...")
        # Interaction features might depend on specific lags being present
        df = create_interaction_features(df, base_cols=config.GROUP_COLS, target_col=config.TARGET_COL, lags=config.LAG_FEATURES_LIST) # Or a subset of lags

    # --- Determine Final Feature List ---
    if config.MODEL_FEATURES:
        features = config.MODEL_FEATURES
    else:
        # Dynamically build feature list based on created features
        features = []
        if config.CREATE_DATE_FEATURES:
            # Infer date feature names (this assumes standard names from create_date_features)
            # A more robust way might be to have create_date_features return the names
            base_date_feats = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'is_month_end', 'is_quarter_end']
            if config.INCLUDE_CYCLICAL_DATE_FEATURES:
                base_date_feats.extend(['month_sin', 'month_cos', 'dow_sin', 'dow_cos'])
            if config.ADD_HOLIDAY_FEATURES:
                 # Check if holiday feature was actually added (depends on package install)
                if 'is_holiday' in df.columns: 
                    base_date_feats.append('is_holiday')
            features.extend(base_date_feats)
            
        if config.CREATE_LAG_FEATURES:
            features.extend([f'{config.TARGET_COL}_lag_{l}' for l in config.LAG_FEATURES_LIST])
        if config.CREATE_ROLLING_FEATURES:
            features.extend([f'{config.TARGET_COL}_roll_{agg}_{w}d' for w in config.ROLLING_WINDOWS for agg in config.ROLLING_AGGS])
        if config.CREATE_EWM_FEATURES:
            features.extend([f'{config.TARGET_COL}_ewm_{s}' for s in config.EWM_SPANS])
        if config.CREATE_INTERACTION_FEATURES:
            features.extend([f'{config.TARGET_COL}_diff_{l}' for l in config.LAG_FEATURES_LIST]) # Assuming diff/ratio based on LAG_FEATURES_LIST
            features.extend([f'{config.TARGET_COL}_ratio_{l}' for l in config.LAG_FEATURES_LIST])
        
        # Add group cols if they should be features themselves (e.g., for categorical)
        features.extend([col for col in config.GROUP_COLS if col not in features])
        # Remove duplicates
        features = sorted(list(set(features)))
        
        # Ensure all generated features actually exist in the dataframe
        features = [f for f in features if f in df.columns]

    logging.info(f"Final features identified: {features}")

    # Drop rows with NaNs created by feature engineering (essential for training)
    initial_rows = len(df)
    if config.TARGET_COL not in df.columns:
        logging.error(f"Target column '{config.TARGET_COL}' not found before dropping NaNs.")
        return None, None
        
    # Drop NaNs based on features AND target
    cols_to_check_for_na = features + [config.TARGET_COL]
    df.dropna(subset=cols_to_check_for_na, inplace=True) 
    rows_after_na_drop = len(df)
    logging.info(f"Dropped {initial_rows - rows_after_na_drop} rows with NaNs after feature engineering.")

    if df.empty:
        logging.error("DataFrame is empty after dropping NaNs. Check lag/rolling features or data span.")
        return None, None
        
    # Sort by date for TimeSeriesSplit
    df = df.sort_values(config.DATE_COL)
    logging.info("Feature engineering complete.")

    # ********** Convert Categorical Features Before CV Split **********
    # Use the final calculated 'features' list
    features_to_use = [f for f in features if f != config.TARGET_COL and f != config.DATE_COL]
    logging.info(f"Using {len(features_to_use)} features for modeling: {features_to_use}")

    # Identify categorical features from config that are in the final feature list
    final_categorical_features = [f for f in config.CATEGORICAL_FEATURES if f in features_to_use]
    logging.info(f"Converting categorical features: {final_categorical_features}")

    # Convert identified categorical columns to 'category' dtype IN THE MAIN DATAFRAME
    for col in final_categorical_features:
        if col in df.columns:
             if df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')
        else:
             logging.warning(f"Configured categorical feature '{col}' not found in the DataFrame.")

    logging.info("Categorical features converted.")
    # ******************************************************************

    # 3. Time Series Cross-Validation
    logging.info("--- Starting Cross-Validation ---")
    tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS) # Use config
    oof_preds = np.zeros(len(df))
    cv_scores = {'mae': [], 'rmse': [], 'smape': []}
    models = [] # Store models from each fold if needed

    X = df[features_to_use]
    y = df[config.TARGET_COL]

    fold = 0
    for train_index, val_index in tscv.split(X):
        fold += 1
        logging.info(f"--- Fold {fold}/{config.CV_FOLDS} ---") # Use config
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if X_train.empty or X_val.empty:
            logging.warning(f"Skipping fold {fold} due to empty train/validation set.")
            continue

        logging.info(f"Train fold size: {len(X_train)}, Validation fold size: {len(X_val)}")

        model = train_lgbm(X_train, y_train, X_val, y_val,
                           features=features_to_use,
                           params=config.LGB_PARAMS) # Pass config params

        val_preds = model.predict(X_val)
        oof_preds[val_index] = val_preds
        models.append(model) # Store model if needed later

        # Evaluate
        mae = mean_absolute_error(y_val, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        smape_score = smape(y_val, val_preds)
        cv_scores['mae'].append(mae)
        cv_scores['rmse'].append(rmse)
        cv_scores['smape'].append(smape_score)
        logging.info(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape_score:.4f}")

    # Calculate average CV scores
    avg_cv_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
    logging.info("--- Cross-Validation Complete ---")
    logging.info(f"Average CV MAE: {avg_cv_scores['mae']:.4f}")
    logging.info(f"Average CV RMSE: {avg_cv_scores['rmse']:.4f}")
    logging.info(f"Average CV SMAPE: {avg_cv_scores['smape']:.4f}")

    # 4. Retrain on Full Data (Optional)
    final_model = None
    if retrain_on_full_data:
        logging.info("--- Retraining on Full Data ---")
        final_params = config.LGB_PARAMS.copy()
        # Potentially increase n_estimators or adjust learning rate for final training

        # X and y already contain the full data with correct dtypes
        X_full = X # Already has features and correct dtypes
        y_full = y

        # No need to convert categories again, they are already set in X_full

        final_model = lgb.LGBMRegressor(**final_params)
        # Fit on the full feature set X_full
        final_model.fit(X_full, y_full,
                        feature_name=features_to_use,
                        categorical_feature='auto') # Let LightGBM detect 'category' dtype
        logging.info("Final model retrained on full data.")
    else:
        # Use the model from the last fold or average models if needed
        logging.info("Skipping retraining on full data. Using model from last CV fold as final model.")
        final_model = models[-1] if models else None

    # 5. Save Model
    if final_model:
        os.makedirs(model_output_dir, exist_ok=True)
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_filename = f"lgbm_model_{timestamp}.joblib"
        model_save_path = os.path.join(model_output_dir, model_filename)
        try:
            joblib.dump(final_model, model_save_path)
            logging.info(f"Final model saved to {model_save_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    else:
        logging.warning("No final model was trained or selected. Skipping model saving.")

    logging.info("--- Training Pipeline Finished ---")
    return final_model, avg_cv_scores

# --- Main Execution ---
def main():
    
    logging.info("Running modeling pipeline directly...")
    
    # Adjust data path relative to where the script is run from (e.g., project root)
    # If run from src/, path is '../data'. If run from root, path is 'data'.
    current_dir = os.path.basename(os.getcwd())
    if current_dir == 'src':
        data_directory = '../data'
        model_directory = '../models'
    else:
        data_directory = 'data'
        model_directory = 'models'
        
    if not os.path.exists(data_directory):
        logging.error(f"Data directory not found at {os.path.abspath(data_directory)}. Please ensure the path is correct relative to the execution location.")
    else:
        trained_model, cv_results = run_training_pipeline(
            data_path=data_directory,
            model_output_dir=model_directory,
            retrain_on_full_data=True
        )
        
        if trained_model:
            logging.info("Pipeline executed successfully.")
            # You could add code here to load the test set, apply features, 
            # make predictions, and generate a submission file.
        else:
            logging.error("Pipeline execution failed.")

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, models: dict[str, any], param_grids: dict[str, dict], cv_splits: int = 5):
    """Train and evaluate multiple models using a time-series pipeline, CV and hyperparameter search."""
    results = {}
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    # Standard preprocessing pipeline
    preprocessor = Pipeline([
        ('scaler', StandardScaler()),
    ])
    for name, model in models.items():
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('estimator', model)
        ])
        grid = RandomizedSearchCV(pipeline, param_grids.get(name, {}), cv=tscv, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)
        grid.fit(X, y)
        # Save best pipeline
        best_model = grid.best_estimator_
        # Evaluate
        preds = best_model.predict(X)
        results[name] = {
            'best_model': best_model,
            'cv_results': grid.cv_results_,
            'MAE': mean_absolute_error(y, preds),
            'RMSE': mean_squared_error(y, preds, squared=False),
            'MAPE': mean_absolute_percentage_error(y, preds)
        }
        # Feature importance if available
        if hasattr(best_model.named_steps['estimator'], 'feature_importances_'):
            results[name]['feature_importances'] = best_model.named_steps['estimator'].feature_importances_
    return results

def ensemble_predictions(results: dict[str, dict], X: pd.DataFrame, weights: dict[str, float] = None) -> np.ndarray:
    """Create a weighted ensemble of model predictions."""
    preds = np.vstack([res['best_model'].predict(X) for res in results.values()]).T
    if not weights:
        weights = {name: 1/len(results) for name in results}
    weighted = np.sum([weights[name] * preds[:, i] for i, name in enumerate(results)], axis=0)
    return weighted 


