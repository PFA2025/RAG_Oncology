from typing import Dict, Any
from src.llm_factory.gemini import GoogleGen
from src.relevance_check.relevance_check import HybridRelevanceChecker
from langchain_core.messages import HumanMessage, SystemMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        self.llm = GoogleGen()
        self.relevance_checker = HybridRelevanceChecker()
        self.system_prompt = """You are an oncology specialist AI. You must:
        - Only answer confirmed cancer-related questions
        - Provide concise, accurate information
        - For unmatched questions, respond:
        "I don't have a verified answer for this oncology question."
        - For non-oncology questions:
        "I only answer cancer-related questions."""

    def generate(self, query: str) -> Dict[str, Any]:
        # Initial oncology check
        if not self.relevance_checker.is_oncology_related(query):
            return {
                'answer': "I only answer cancer-related questions.",
                'source': 'filtered',
                'confidence': 1.0
            }
        
        # Check for direct matches
        match_result = self.relevance_checker.check_match(query)
        
        # Return direct match if exists
        if match_result['status'] == 'direct_match':
            return {
                'answer': match_result['match_data']['answer'],
                'source': 'verified_answer',
                'confidence': match_result['match_data']['confidence']
            }
        
        # For no-match oncology questions
        if match_result['status'] == 'no_match':
            return {
                'answer': "I don't have a verified answer for this oncology question.",
                'source': 'no_match',
                'confidence': 0.0
            }
        
        # Fallback (shouldn't reach here with current logic)
        return {
            'answer': "I can't provide an answer at this time.",
            'source': 'error',
            'confidence': 0.0
        }