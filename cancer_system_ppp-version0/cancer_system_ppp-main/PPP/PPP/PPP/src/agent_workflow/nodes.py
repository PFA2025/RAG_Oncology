import sys
import logging
from datetime import datetime
from helpers.document_retriever import *
from llm_factory.gemini import GoogleGen
from relevance_check.relevance_check import HybridRelevanceChecker
from answer_generator.answer_generator import AnswerGenerator
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import time
import os
from pathlib import Path
import json
import uuid
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

class Nodes:
    def __init__(self):
        """Initialize nodes with necessary components."""
        try:
            self.llm_obj = GoogleGen()
            self.tools = []
            self.llm_obj.llm_with_tools = self.llm_obj.llm.bind_tools(self.tools)
            self.max_retries = 3
            self.privacy_keywords = ['personal', 'private', 'confidential', 'sensitive']
            self.relevance_checker = HybridRelevanceChecker()
            self.answer_generator = AnswerGenerator()
            logger.info("Nodes initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing nodes: {str(e)}")
            raise

    def _sanitize_input(self, text: str) -> str:
        """Remove or mask potentially sensitive information."""
        try:
            for keyword in self.privacy_keywords:
                if keyword in text.lower():
                    text = text.replace(keyword, '[REDACTED]')
            return text
        except Exception as e:
            logger.error(f"Error sanitizing input: {str(e)}")
            return text

    def initiate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the conversation state with metadata."""
        try:
            session_id = str(uuid.uuid4())
            return {
                'session_id': session_id,
                'timestamp': datetime.now(),
                'privacy_level': 'high',  # Default to high privacy
                'metadata': {
                    'start_time': datetime.now().isoformat(),
                    'version': '1.0',
                    'relevance_checks': [],
                    'answer_sources': []
                },
                'error_count': 0,
                'last_interaction': datetime.now(),
                'messages': []
            }
        except Exception as e:
            logger.error(f"Error in initiate_state: {str(e)}")
            raise

    def prepare_prompt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the system prompt with guidelines and privacy notices."""
        try:
            logger.info('Preparing system prompt')
            with open(os.path.abspath(os.path.join(current_dir, "..", "prompts/guidelines.txt")), "r") as file:
                guidelines = file.read()

            privacy_notice = """
            IMPORTANT: This conversation is private and confidential. 
            All personal information will be handled with strict privacy measures.
            """

            system_prompt = f"""
            {guidelines}
            
            {privacy_notice}
            
            Remember to:
            1. Always prioritize patient privacy
            2. Use clear, simple language
            3. Be empathetic and understanding
            4. Recommend professional consultation when needed
            5. Acknowledge uncertainty when present
            6. Ensure answers are relevant and accurate
            """

            return {'messages': [SystemMessage(content=system_prompt)]}
        except Exception as e:
            logger.error(f"Error in prepare_prompt: {str(e)}")
            state['error_count'] += 1
            raise

    def user(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user input with privacy measures and relevance checking."""
        try:
            logger.info('Processing user input')
            
            # Initialize metadata if not present
            if 'metadata' not in state:
                state['metadata'] = {
                    'relevance_checks': [],
                    'start_time': datetime.now().isoformat(),
                    'version': '1.0',
                    'answer_sources': []
                }
            
            # Get the last message from state
            if not state.get('messages'):
                return {'messages': [HumanMessage(content="I didn't catch that. Could you please repeat your question?")]}
                
            last_message = state['messages'][-1]
            if not isinstance(last_message, HumanMessage):
                return {'messages': [HumanMessage(content="I didn't catch that. Could you please repeat your question?")]}
                
            user_input = last_message.content
            if not user_input.strip():
                return {'messages': [HumanMessage(content="I didn't catch that. Could you please repeat your question?")]}

            # Sanitize input
            sanitized_input = self._sanitize_input(user_input)
            
            # Check relevance and get answer
            answer_result = self.answer_generator.generate(sanitized_input)
            
            # Store answer result in state
            state['answer_result'] = answer_result
            
            # Update state with relevance check results
            if 'relevance_checks' not in state['metadata']:
                state['metadata']['relevance_checks'] = []
                
            state['metadata']['relevance_checks'].append({
                'timestamp': datetime.now().isoformat(),
                'query': sanitized_input,
                'status': answer_result.get('source', 'unknown'),
                'confidence': answer_result.get('confidence', 0.0)
            })
            
            # Update last interaction time
            state['last_interaction'] = datetime.now()
            
            # Return updated state
            return state
            
        except Exception as e:
            logger.error(f"Error in user node: {str(e)}")
            state['error_state'] = True
            state['messages'].append(AIMessage(content="I apologize, but I encountered an error. Please try again."))
            return state

    def agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent response with error handling and privacy checks."""
        try:
            logger.info('Generating agent response')
            if state.get('error_state'):
                return state

            # Get the answer result from state
            answer_result = state.get('answer_result', {})
            if not answer_result:
                state['error_state'] = True
                state['messages'].append(AIMessage(content="I apologize, but I couldn't process your question. Please try again."))
                return state
            
            # Sanitize response for privacy
            sanitized_response = self._sanitize_input(answer_result['answer'])
            
            # Add AI response to messages
            state['messages'].append(AIMessage(content=sanitized_response))
            
            return state
            
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            state['error_state'] = True
            state['messages'].append(AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again."))
            return state

    def final_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversation end and cleanup."""
        try:
            logger.info(f"Ending session {state['session_id']}")
            # Add session summary to metadata
            state['metadata']['end_time'] = datetime.now().isoformat()
            state['metadata']['total_messages'] = len(state['messages'])
            state['metadata']['relevance_stats'] = {
                'total_checks': len(state['metadata']['relevance_checks']),
                'verified_answers': sum(1 for check in state['metadata']['relevance_checks'] 
                                     if check['status'] == 'verified_answer'),
                'average_confidence': sum(check['confidence'] for check in state['metadata']['relevance_checks']) 
                                    / len(state['metadata']['relevance_checks']) 
                                    if state['metadata']['relevance_checks'] else 0
            }
            return state
        except Exception as e:
            logger.error(f"Error in final_state: {str(e)}")
            return state