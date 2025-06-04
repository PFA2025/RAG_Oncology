from typing import Dict, List, Any
from llm_factory.gemini import GoogleGen
from relevance_check.relevance_check import HybridRelevanceChecker
from langchain_core.messages import HumanMessage, SystemMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        self.llm = GoogleGen()
        self.relevance_checker = HybridRelevanceChecker()
    
    def generate(self, query: str) -> Dict[str, Any]:
        relevance_result = self.relevance_checker.check_match(query)
        
        if relevance_result['status'] == 'relevant':
            return {
                'answer': relevance_result['match_data']['answer'],
                'source': 'verified_answer',
                'confidence': relevance_result['match_data']['confidence'],
                'metrics': relevance_result['match_data']['metrics']
            }
        
        messages = [
            SystemMessage(content="You are an expert oncology assistant."),
            HumanMessage(content=query)
        ]
        
        if relevance_result['status'] == 'partial':
            messages.insert(1, HumanMessage(
                content=f"Context:\n{relevance_result['match_data']['answer']}"
            ))
        
        try:
            response = self.llm(messages)
            return {
                'answer': response.content,
                'source': 'llm_generated',
                'confidence': 0.7 if relevance_result['status'] == 'partial' else 0.5
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I'm sorry, I couldn't generate a response. Please try again later.",
                'source': 'error',
                'confidence': 0.0
            }