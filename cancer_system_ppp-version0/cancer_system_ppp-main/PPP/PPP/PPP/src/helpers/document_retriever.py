from langchain_chroma import Chroma
from langchain.schema import Document
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / '../../data/data_oncology.xlsx'
VECTOR_STORE_DIR = SCRIPT_DIR / '../../chroma_db_oncology'

# Initialize models
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

def remove_duplicates(df: pd.DataFrame, similarity_threshold: float = 0.85) -> pd.DataFrame:
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
    
    oncology_data = remove_duplicates(oncology_data)
    
    documents = []
    for _, row in oncology_data.iterrows():
        content = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
        metadata = {"Question": row['Question'], "Answer": row['Answer']}
        documents.append(Document(page_content=content, metadata=metadata))
    
    vector_store.add_documents(documents=documents)
    logger.info(f"Vector store created with {len(documents)} documents.")
    return vector_store

def search_qa(query: str, k: int = 5, use_cross_encoder: bool = True) -> List[Dict[str, Any]]:
    try:
        logger.info(f"Searching knowledge base for: {query}")
        
        embeddings = SentenceTransformerEmbeddings(bi_encoder)
        vector_store = Chroma(
            collection_name="oncology_qa",
            embedding_function=embeddings,
            persist_directory=str(VECTOR_STORE_DIR)
        )
        
        initial_results = vector_store.similarity_search(query, k=k*3 if use_cross_encoder else k)
        
        if not initial_results:
            return []
        
        if not use_cross_encoder:
            return [{
                "question": doc.metadata.get('Question'),
                "answer": doc.metadata.get('Answer'),
                "score": 1.0
            } for doc in initial_results[:k]]
        
        unique_pairs = [(query, doc.page_content) for doc in initial_results]
        scores = cross_encoder.predict(unique_pairs)
        
        scored_results = list(zip(initial_results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [{
            "question": doc.metadata.get('Question'),
            "answer": doc.metadata.get('Answer'),
            "score": float(score)
        } for doc, score in scored_results[:k]]
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}")
        return []