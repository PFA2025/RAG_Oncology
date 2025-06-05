import logging
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import shutil

from .constants import VECTOR_STORE_DIR, DATA_FILE
from .document_retriever import embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vectorstore():
    """Create and initialize the FAISS vector store from the oncology Q&A data."""
    load_dotenv()
    
    # Remove existing FAISS index if it exists
    if VECTOR_STORE_DIR.exists():
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
            logger.info(f"Removed existing FAISS index at {VECTOR_STORE_DIR}")
        except Exception as e:
            logger.warning(f"Error removing existing FAISS index: {e}")
    
    # Create parent directory if it doesn't exist
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        if not DATA_FILE.exists():
            logger.error(f"Data file not found at: {DATA_FILE}")
            return None
            
        # Load and clean data
        oncology_data = pd.read_excel(DATA_FILE)
        logger.info(f"Loaded {len(oncology_data)} rows from Excel")
        
        # Remove exact duplicates
        oncology_data = oncology_data.drop_duplicates(subset=['Question', 'Answer'])
        logger.info(f"After removing exact duplicates: {len(oncology_data)} rows")
        
        # Create documents with metadata
        documents = []
        for idx, row in oncology_data.iterrows():
            content = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
            metadata = {
                "source": str(idx), 
                "question": row['Question'], 
                "answer": row['Answer']
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create and save the FAISS index
        logger.info(f"Creating FAISS index with {len(documents)} documents...")
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        
        # Save the index
        vector_store.save_local(
            folder_path=str(VECTOR_STORE_DIR),
            index_name="oncology_qa"
        )
        
        logger.info(f"FAISS index created and saved to {VECTOR_STORE_DIR}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        return None

def main():
    logger.info("Initializing vector store...")
    vector_store = create_vectorstore()
    if vector_store:
        logger.info("Vector store initialized successfully")
        return vector_store
    else:
        logger.error("Failed to initialize vector store")
        return None

if __name__ == "__main__":
    main()