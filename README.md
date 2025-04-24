# Store Item Demand Forecasting

This project aims to forecast the demand for items across different stores using historical sales data.

## Project Structure

```
.
├── app/                # Streamlit application code
│   └── main.py         # Main application script (to be created)
├── data/               # Raw and processed data
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/          # Jupyter notebooks for EDA and experimentation
│   └── eda.ipynb       # Exploratory Data Analysis (to be created)
├── src/                # Source code for data processing, feature engineering, modeling
│   ├── __init__.py
│   ├── data_processing.py (to be created)
│   ├── feature_engineering.py (to be created)
│   └── modeling.py      (to be created)
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd store_item_demand_forecasting_cursor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Build and run with Docker:**
    ```bash
    docker build -t store-demand-forecasting .
    docker run -p 8501:8501 store-demand-forecasting
    ```

## Usage

1.  **Run EDA notebook:** Navigate to the `notebooks` directory and run `jupyter notebook eda.ipynb`.
2.  **Train models:** (Details on running training scripts TBD)
3.  **Run Streamlit app:**
    ```bash
    streamlit run app/main.py
    ```
    Or access via Docker at `http://localhost:8501`. 