from typing import Dict, Any, List
from helpers.document_retriever import search_qa
from llm_factory.gemini import GoogleGen
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage
import numpy as np
import json
import re
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRelevanceChecker:
    def __init__(self):
        """Simplified relevance checker for oncology with only direct matches"""
        self.llm = GoogleGen()
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Thresholds
        self.similarity_threshold = 0.8
        self.confidence_threshold = 0.85

    def is_oncology_related(self, text: str) -> bool:
        """Strict oncology content check"""
        prompt = """Is this text about cancer/oncology? Answer ONLY 'yes' or 'no'.
        
        Text: {text}""".format(text=text)
        
        try:
            response = self.llm([HumanMessage(content=prompt)])
            return response.content.strip().lower() == 'yes'
        except Exception as e:
            logger.error(f"Oncology check failed: {e}")
            return False

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between texts"""
        embeds = self.similarity_model.encode([text1, text2])
        return float(np.dot(embeds[0], embeds[1]) / 
                   (np.linalg.norm(embeds[0]) * np.linalg.norm(embeds[1])))

    def verify_match(self, query: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Verify if candidate is a direct match"""
        similarity = self.calculate_similarity(query, candidate['question'])
        
        verification_prompt = """Verify if this answer perfectly matches the question.
        Question: {query}
        Answer: {answer}
        
        Return JSON with:
        - match (true/false)
        - confidence (0.0-1.0)
        - reason (brief explanation)""".format(
            query=query,
            answer=candidate['answer']
        )
        
        try:
            response = self.llm([HumanMessage(content=verification_prompt)])
            verification = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())
            
            return {
                'similarity': similarity,
                'verification': verification,
                'combined_score': min(1.0, (similarity + verification['confidence']) / 2)
            }
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'similarity': similarity,
                'verification': {'match': False, 'confidence': 0.0, 'reason': str(e)},
                'combined_score': 0.0
            }

    def check_match(self, query: str) -> Dict[str, Any]:
        """Check for direct matches only"""
        if not self.is_oncology_related(query):
            return {'status': 'off_topic', 'match_data': None}
        
        rag_results = search_qa(query=query, k=5)  # Fewer but more relevant results
        if not rag_results:
            return {'status': 'no_match', 'match_data': None}
        
        # Evaluate all candidates
        evaluations = []
        for candidate in rag_results:
            eval_result = self.verify_match(query, candidate)
            if eval_result['verification']['match'] and eval_result['combined_score'] >= self.confidence_threshold:
                evaluations.append({
                    'candidate': candidate,
                    'metrics': eval_result,
                    'confidence': eval_result['combined_score']
                })
        
        if not evaluations:
            return {'status': 'no_match', 'match_data': None}
        
        # Return best direct match
        best_match = max(evaluations, key=lambda x: x['confidence'])
        return {
            'status': 'direct_match',
            'match_data': {
                'answer': best_match['candidate']['answer'],
                'confidence': best_match['confidence'],
                'metrics': best_match['metrics']
            }
        }