"""Constants and shared configurations for the helpers module."""
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Initialize models
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Path setup
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / '../../data/data_oncology.xlsx'
VECTOR_STORE_DIR = SCRIPT_DIR / '../../chroma_db_oncology'
