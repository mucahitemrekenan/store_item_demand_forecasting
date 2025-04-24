# Contents for Forecast tab
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging

# Import necessary functions directly (ensure PYTHONPATH allows this)
try:
    from src.feature_engineering import create_features_for_prediction
except ImportError:
    st.error("Could not import feature engineering functions for forecasting.")
    # Define a dummy if import fails, though this tab will likely be broken
    def create_features_for_prediction(*args, **kwargs):
        st.warning("Using dummy create_features_for_prediction.")
        return None

def render_forecast_tab(model, historical_data, selected_store, selected_item, FORECAST_HORIZON):
    """Renders the Forecast tab content."""
    with st.container():
        st.subheader("Sales Forecast")
        
        if model is None:
            st.warning("Please select and load a model before generating forecasts.")
        elif historical_data.empty:
            st.warning("No historical data available for the selected store/item/date range to base the forecast on.")
        else:
            # Button is now inside the conditionally rendered container
            if st.button("Generate Forecast", key="generate_forecast_button"):
                with st.spinner("Generating forecast..."):
                    try:
                        last_known_date = historical_data['date'].max()
                        future_idx = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='D')
                        
                        # Use the full historical data for feature generation, not just the displayed range?
                        # Pass the filtered historical_data for context
                        features_for_pred = create_features_for_prediction(
                            historical_data, # Use filtered history for context
                            future_idx, 
                            selected_store, 
                            selected_item
                        )

                        if features_for_pred is None:
                            st.error("Failed to generate features for prediction. Check logs.")
                        else:
                            # Ensure columns align exactly with model's expected features
                            try:
                                # Ensure feature order matches model.feature_name_
                                feats_model_input = features_for_pred[model.feature_name_] 
                            except KeyError as e:
                                 st.error(f"Feature mismatch: Model expects feature '{e}' which was not generated for prediction. Check feature engineering steps.")
                                 st.stop() # Stop if features don't match
                            except AttributeError:
                                st.error("Loaded model object does not have 'feature_name_'. Is it a trained LightGBM model?")
                                st.stop()

                            preds = model.predict(feats_model_input)
                            # Post-process predictions (clip negatives, round)
                            preds = np.clip(np.round(preds), 0, None).astype(int) 
                            
                            forecast_df = pd.DataFrame({'date': future_idx, 'forecast': preds})
                            
                            # Display forecast plot
                            history_to_plot = historical_data.tail(min(365, len(historical_data))) 
                            fig_f = px.line(history_to_plot, x='date', y='sales', title="History & Forecast")
                            fig_f.add_scatter(x=forecast_df['date'], y=forecast_df['forecast'], mode='lines', name='Forecast', line=dict(color='orange'))
                            st.plotly_chart(fig_f, use_container_width=True)
                            
                            # Display forecast table
                            st.dataframe(forecast_df, use_container_width=True)
                            
                            # Download button
                            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Forecast CSV",
                                data=csv_data,
                                file_name=f"forecast_store{selected_store}_item{selected_item}_{future_idx.min().strftime('%Y%m%d')}_{future_idx.max().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")
                        logging.error(f"Forecast generation error: {e}")
            else:
                 st.info("Click 'Generate Forecast' to run the selected model.") 