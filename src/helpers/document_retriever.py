from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Import from constants to share models and paths
from .constants import VECTOR_STORE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


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

def get_vector_store() -> FAISS:
    """Initialize and return the FAISS vector store."""
    # Check if the index exists
    index_path = VECTOR_STORE_DIR / "oncology_qa.faiss"
    
    if index_path.exists():
        try:
            # Load existing index with deserialization allowed
            return FAISS.load_local(
                folder_path=str(VECTOR_STORE_DIR),
                embeddings=embeddings,
                index_name="oncology_qa",
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise RuntimeError("Failed to load existing FAISS index")
    else:
        # Create a new empty index with a dummy document
        logger.warning("No existing FAISS index found, creating a new one")
        return FAISS.from_texts(
            texts=["Initial document"],
            embedding=embeddings,
        )

def search_qa(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the QA knowledge base for relevant answers.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List of dictionaries containing question, answer, and score
    """
    try:
        logger.info(f"Searching knowledge base for: {query}")
        
        # Get the vector store
        logger.info("Loading vector store...")
        vector_store = get_vector_store()
        
        # Use similarity search with scores
        logger.info("Performing similarity search...")
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Found {len(docs_with_scores)} results")
        
        if not docs_with_scores:
            logger.warning("No results found for query")
            return []
            
        # Format results
        results = []
        for doc, score in docs_with_scores:
            doc.metadata['score'] = float(score)
            results.append(format_result(doc))
            
        return results
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}", exc_info=True)
        return []