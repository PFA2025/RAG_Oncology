from langchain_chroma import Chroma
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Import from constants to share models and paths
from .constants import bi_encoder, VECTOR_STORE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize cross-encoder for re-ranking
cross_encoder = None

def get_cross_encoder():
    """Lazy load the cross-encoder to avoid loading it if not needed."""
    global cross_encoder
    if cross_encoder is None:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return cross_encoder

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()


def format_result(doc: Document) -> Dict[str, Any]:
    """Format a document into a result dictionary."""
    # Parse the question and answer from the page_content
    content = doc.page_content
    question = ""
    answer = ""
    
    # The content is in format "Question: ...\nAnswer: ..."
    if "Question:" in content and "Answer:" in content:
        parts = content.split("Answer:", 1)
        question = parts[0].replace("Question:", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
    
    return {
        "question": question,
        "answer": answer,
    }

def get_vector_store() -> Chroma:
    """Initialize and return the Chroma vector store."""
    embeddings = SentenceTransformerEmbeddings(bi_encoder)
    return Chroma(
        collection_name="oncology_qa",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )

def search_qa(query: str, k: int = 5, use_cross_encoder: bool = False) -> List[Dict[str, Any]]:
    """
    Search the QA knowledge base for relevant answers.
    
    Args:
        query: The search query
        k: Number of results to return
        use_cross_encoder: Whether to use cross-encoder for re-ranking
        
    Returns:
        List of dictionaries containing question, answer, and score
    """
    try:
        
        logger.info(f"Searching knowledge base for: {query}")
        
        vector_store = get_vector_store()
        docs = vector_store.get()
        logger.info(f"printing all the doc that exist in vector store")
        logger.info('---------------------------------')
        for i, doc in enumerate(docs['documents']):
            print(f"\nDocument {i+1}:")
            print(doc)
        logger.info('---------------------------------')
        fetch_count = k * 3 if use_cross_encoder else k
        initial_results = vector_store.similarity_search(query, k=fetch_count)
        
        if not initial_results:
            return []
            
        if not use_cross_encoder:
            return [format_result(doc) for doc in initial_results[:k]]
        
        
        # Re-rank with cross-encoder
        query_doc_pairs = [(query, doc.page_content) for doc in initial_results]
        cross_encoder = get_cross_encoder()
        scores = cross_encoder.predict(query_doc_pairs)
        
        # Combine results with scores and sort
        scored_results = zip(initial_results, scores)
        top_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:k]
        
        return [format_result(doc) for doc in top_results]
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}", exc_info=True)
        return []