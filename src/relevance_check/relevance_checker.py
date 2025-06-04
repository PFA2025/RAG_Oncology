from typing import Dict, Any, List, Optional
from sentence_transformers import CrossEncoder, util
import numpy as np
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RelevanceResult:
    """Container for relevance check results."""
    is_relevant: bool
    confidence: float
    best_match: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

class RelevanceChecker:
    """
    A class to check the relevance of search results to a given query.
    Uses semantic similarity and cross-encoder for more accurate matching.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, relevance_threshold: float = 0.7):
        """
        Initialize the RelevanceChecker.
        
        Args:
            similarity_threshold: Minimum similarity score to consider a match
            relevance_threshold: Minimum confidence score to consider a result relevant
        """
        self.similarity_threshold = similarity_threshold
        self.relevance_threshold = relevance_threshold
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def _calculate_similarity(self, query: str, text: str) -> float:
        """Calculate semantic similarity between query and text."""
        try:
            # Use cross-encoder for more accurate similarity scoring
            pairs = [[query, text]]
            scores = self.cross_encoder.predict(pairs)
            return float(scores[0])
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def check_relevance(self, query: str, search_results: List[Dict[str, Any]]) -> RelevanceResult:
        """
        Check the relevance of search results to the given query.
        
        Args:
            query: The user's query
            search_results: List of search results from the document retriever
            
        Returns:
            RelevanceResult containing relevance information
        """
        if not search_results:
            return RelevanceResult(
                is_relevant=False,
                confidence=0.0,
                reason="No search results available"
            )
        
        best_match = None
        best_score = -1
        
        for result in search_results:
            try:
                # Calculate similarity between query and result
                similarity = self._calculate_similarity(query, result['question'])
                
                # Update best match if current score is higher
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'question': result['question'],
                        'answer': result['answer'],
                        'score': result.get('score', 0.0),
                        'similarity': similarity
                    }
            except Exception as e:
                logger.error(f"Error processing search result: {e}")
                continue
        
        # Determine if the best match is relevant
        is_relevant = best_score >= self.similarity_threshold
        
        return RelevanceResult(
            is_relevant=is_relevant,
            confidence=best_score,
            best_match=best_match,
            reason="Relevant result found" if is_relevant else "No sufficiently relevant results found"
        )

# Global instance for easy import
relevance_checker = RelevanceChecker()

def check_relevance(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check the relevance of search results to the given query.
    
    This is a convenience function that uses the global relevance_checker instance.
    
    Args:
        query: The user's query
        search_results: List of search results from the document retriever
        
    Returns:
        Dictionary containing relevance information
    """
    result = relevance_checker.check_relevance(query, search_results)
    return {
        'is_relevant': result.is_relevant,
        'confidence': result.confidence,
        'best_match': result.best_match,
        'reason': result.reason
    }
