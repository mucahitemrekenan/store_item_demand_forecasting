# Contents for Model Info tab
import streamlit as st
import pandas as pd
import plotly.express as px

def render_model_info_tab(model):
    """Renders the Model Info tab content."""
    with st.container():
        st.subheader("Model Information")
        if model is None:
             st.warning("No model selected or loaded. Cannot display model information.")
        else:
            # Display model parameters (example)
            try:
                st.write("Model Parameters:")
                # Convert model params to a more readable format if needed
                params = model.get_params()
                st.json(params, expanded=False) # Display params collapsed
            except Exception as e:
                st.warning(f"Could not display model parameters: {e}")

            # Display feature importances
            try:
                # Check if feature names are available (LightGBM stores them)
                if hasattr(model, 'feature_name_') and hasattr(model, 'feature_importances_'):
                    fi = pd.DataFrame({'feature': model.feature_name_, 'importance': model.feature_importances_})
                    fi = fi.sort_values('importance', ascending=False).head(20)
                    fig_imp = px.bar(fi, x='importance', y='feature', orientation='h', title="Top 20 Feature Importances")
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importances are not available for this model type or model is not trained.")
            except Exception as e:
                st.warning(f"Could not display feature importances: {e}") 