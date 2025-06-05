"""Constants and shared configurations for the helpers module."""
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / '../../data/data_oncology.xlsx'
VECTOR_STORE_DIR = SCRIPT_DIR / '../../faiss_index'
