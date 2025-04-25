# Store Item Demand Forecasting

This project aims to forecast the demand for items across different stores using historical sales data.

## Project Structure

```
.
├── apps/               # Streamlit application code (if any)
│   └── streamlit_app/  # Specific Streamlit app code
│       ├── __init__.py
│       ├── streamlit_dashboard.py
│       ├── sidebar.py
│       ├── forecast_tab.py
│       ├── features_tab.py
│       ├── model_info_tab.py
│       └── eda_tab.py
├── data/               # Raw and processed data (e.g., train.csv, test.csv)
    ├── raw
    └── processed
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for EDA and experimentation
    └── eda_script.ipynb
├── requirements.txt    # Python dependencies
├── run.py              # Main script to run pipelines (e.g., training, prediction)
├── src/                # Source code for data processing, features, modeling, pipelines
│   ├── __init__.py
│   ├── config.py       # Configuration file for parameters and paths
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── Dockerfile          # Docker configuration
└── README.md           # Project overview
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd store_item_demand_forecasting # Or your project directory name
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\\Scripts\\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Build and run with Docker:**
    ```bash
    docker build -t store-demand-forecasting .
    # Add appropriate volume mounts if needed for data/models
    docker container run -p 8501:8501 store-demand-forecasting  # Or run a specific command
    ```

## Usage

1.  **Configure model and feature engineering parameters:** Modify `src/config.py` as needed.
2.  **Run pipelines:** Execute the main script `run.py`. Be aware that you run the command in the project root.
    Example:
    ```bash
    @root_path_of_the_project> streamlit run run.py
    ```
3.  **Explore notebooks:** Navigate to the `notebooks` directory and run Jupyter notebooks for analysis.
    ```bash
    jupyter notebook
    ```
