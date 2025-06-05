import logging
import pandas as pd
import numpy as np
from pathlib import Path
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

from .constants import bi_encoder, VECTOR_STORE_DIR, DATA_FILE, SCRIPT_DIR
from .document_retriever import SentenceTransformerEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _remove_duplicates(df: pd.DataFrame, similarity_threshold: float = 0.85) -> pd.DataFrame:
    logger.info("Removing duplicates from dataset")
    print(f"Initial number of entries: {len(df)}")
    
    # 1. Remove exact duplicates
    df = df.drop_duplicates(subset=['Question', 'Answer'], keep='first')
    print(f"After removing exact duplicates: {len(df)}")
    
    # 2. Remove similar questions
    if len(df) > 1:
        questions = df['Question'].tolist()
        question_embeddings = bi_encoder.encode(questions)
        similarity_matrix = np.dot(question_embeddings, question_embeddings.T)
        
        to_drop = set()
        for i in range(len(df)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(df)):
                if j in to_drop:
                    continue
                if similarity_matrix[i, j] > similarity_threshold:
                    if len(df.iloc[i]['Answer']) < len(df.iloc[j]['Answer']):
                        to_drop.add(i)
                    else:
                        to_drop.add(j)
        
        df = df.drop(df.index[list(to_drop)])
        print(f"After removing similar questions: {len(df)}")
    
    return df


def create_vectorstore():
    load_dotenv()
    embeddings = SentenceTransformerEmbeddings(bi_encoder)
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        client.delete_collection("oncology_qa")
        logger.info("Deleted existing collection")
    except Exception as e:
        logger.info(f"No existing collection to delete: {e}")
    
    vector_store = Chroma(
        collection_name="oncology_qa",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    
    try:
        if not DATA_FILE.exists():
            logger.error(f"Data file not found at: {DATA_FILE}")
            return None
            
        oncology_data = pd.read_excel(DATA_FILE)
        logger.info(f"Loaded {len(oncology_data)} rows from Excel")
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return None
    
    oncology_data = _remove_duplicates(oncology_data)
    
    documents = []
    for _, row in oncology_data.iterrows():
        content = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
        documents.append(Document(page_content=content))
    
    vector_store.add_documents(documents=documents)
    logger.info(f"Vector store created with {len(documents)} documents.")
    return vector_store

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