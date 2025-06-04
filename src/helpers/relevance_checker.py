from typing import Dict, Any, List
from langchain.schema import HumanMessage
from llm_factory.gemini import GoogleGen

def check_relevance(query: str, search_result: Dict[str, Any], llm=None) -> bool:
    """
    Check if any search results are relevant to the query using LLM.
    
    Args:
        query: The user's query
        search_result: List of search results from the document retriever
        llm: Optional LLM instance (defaults to GoogleGen)
        
    Returns:
        bool
    """
    if not search_result:
        return False
    
    llm = llm or GoogleGen()
    
    try:
        prompt = f"""You are a medical information assistant. 
        Determine if the following text is relevant to the user's query.
        
        Query: {query}
        
        Text: {search_result['question']}
            
        Respond with ONLY 'yes' or 'no'."""
            
        response = llm([HumanMessage(content=prompt)])
        if response.content.strip().lower().startswith('yes'):
            return True
    except Exception as e:
        print(f"Error checking relevance: {e}")
        
    return False
