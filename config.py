import pathlib
import os # Optional: Only needed if you want to read environment variables as overrides

# Define the absolute path to the directory containing this config file
# __file__ is the path to the current script (config.py)
# .resolve() makes the path absolute
# .parent gets the directory containing the file
CONFIG_DIR = pathlib.Path(__file__).resolve().parent

# --- Core Path Definitions ---

# Define the Project Root directory
# If config.py is at the root, ROOT_DIR is the same as CONFIG_DIR
ROOT_DIR = CONFIG_DIR
# If config.py were inside a 'src' or 'conf' folder, you might use:
# ROOT_DIR = CONFIG_DIR.parent # Moves one level up from config's directory

# Define other paths relative to the ROOT_DIR
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src" # Optional: if you need to reference the source directory

# --- Specific Sub-paths (Optional but often useful) ---
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FORECAST_HORIZON = 90 # Days to forecast into the future (matches test set duration)

# --- Optional: Environment Variable Overrides ---
# Uncomment if you want the flexibility to override paths via environment variables
# This is useful for CI/CD or specific deployment scenarios, but adds complexity.
# ROOT_DIR = pathlib.Path(os.getenv("PROJECT_ROOT_DIR", ROOT_DIR)).resolve()
# DATA_DIR = pathlib.Path(os.getenv("PROJECT_DATA_DIR", DATA_DIR)).resolve()
# MODELS_DIR = pathlib.Path(os.getenv("PROJECT_MODELS_DIR", MODELS_DIR)).resolve()
# --- Make sure overridden paths exist if needed ---
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --- Example Usage Print (for verification when running config.py directly) ---
if __name__ == "__main__":
    print(f"Configuration loaded from: {__file__}")
    print(f"Project Root Directory: {ROOT_DIR}")
    print(f"Data Directory:         {DATA_DIR}")
    print(f"Models Directory:       {MODELS_DIR}")
    print(f"Source Directory:       {SRC_DIR}")
    print(f"Raw Data Directory:     {RAW_DATA_DIR}")
    print(f"Processed Data Dir:   {PROCESSED_DATA_DIR}")

    # Example check for directory existence
    print(f"\nChecking existence:")
    print(f"Data dir exists:    {DATA_DIR.exists()}")
    print(f"Models dir exists:  {MODELS_DIR.exists()}")

    # Create directories if they don't exist (optional here, maybe better in setup/init script)
    # DATA_DIR.mkdir(parents=True, exist_ok=True)
    # MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)