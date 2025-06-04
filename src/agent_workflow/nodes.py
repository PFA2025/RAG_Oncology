import sys
import logging
from datetime import datetime
from llm_factory.gemini import GoogleGen
from helpers.relevance_checker import *
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.schema import Document
from helpers.document_retriever import *

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
            logger.info("Nodes initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing nodes: {str(e)}")
            raise

    def initiate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the conversation state"""
        logger.info(f"Initializing conversation state")
        return state
        # try:
        #     bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        #     logger.info(f"Searching knowledge base for: {state['user_input']}")
            
        #     embeddings = SentenceTransformerEmbeddings(bi_encoder)
        #     vector_store = Chroma(
        #         collection_name="oncology_qa",
        #         embedding_function=embeddings,
        #         persist_directory=str(VECTOR_STORE_DIR)
        #     )
            
        #     # initial_results = vector_store.similarity_search(state['user_input'], k=k*3 if use_cross_encoder else k)
        #     initial_results = vector_store.similarity_search(state['user_input'], k=5)
        #     if not initial_results:
        #         return []
            
        #     # if not use_cross_encoder:
        #     #     return [{
        #     #         "question": doc.metadata.get('Question'),
        #     #         "answer": doc.metadata.get('Answer'),
        #     #         "score": 1.0
        #     #     } for doc in initial_results[:k]]
            
        #     unique_pairs = [(state['user_input'], doc.page_content) for doc in initial_results]
        #     scores = cross_encoder.predict(unique_pairs)
            
        #     scored_results = list(zip(initial_results, scores))
        #     scored_results.sort(key=lambda x: x[1], reverse=True)
            
        #     return [{
        #         "question": doc.metadata.get('Question'),
        #         "answer": doc.metadata.get('Answer'),
        #         "score": float(score)
        #     } for doc, score in scored_results[:k]]
        
        # except Exception as e:
        #     logger.error(f"Search failed for state['user_input'] '{state['user_input']}': {str(e)}")
        #     return []
    
    def document_retriever(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Document retriever"""
        logger.info(f"Running document retriever")
        try:
            query = state["user_input"]
            state["search_results"] = search_qa(query)
            return state
        
        except Exception as e:
            logger.error(f"Error in document retriever: {str(e)}")
            state["error_state"] = True
            state["messages"].append(
                AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again.")
            )
            return state
            
    def relevance_checker(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """relevance_checker"""  
        logger.info(f"Running relevance checker")
        try:
            for result in state["search_results"]:
                result["is_relevant"] = check_relevance(state["user_input"], result)
            
            for result in state["search_results"]:
                if not result["is_relevant"]:
                    state["search_results"].remove(result)
            return state
            
        except Exception as e:
            logger.error(f"Error in relevance checker: {str(e)}")
            state["error_state"] = True
            state["messages"].append(
                AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again.")
            )
            
            return state
            
    def prepare_prompt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the system prompt with guidelines and privacy notices."""
        logger.info(f"Preparing system prompt")
        try:
            logger.info('Preparing system prompt')
            with open(os.path.abspath(os.path.join(current_dir, "..", "prompts/guidelines.txt")), "r") as file:
                guidelines = file.read()

            system_prompt = f"""
            {guidelines}
            """
            human_prompt="""
            This is the user prompt: {user_prompt}
            This is the retrieved data: {retrieved_data}
            """.format(user_prompt=state['user_prompt'],retrieved_data=state['retrieved_data'])

            return {'messages': [SystemMessage(content=system_prompt),HumanMessage(content=system_prompt)]}
        except Exception as e:
            logger.error(f"Error in prepare_prompt: {str(e)}")
            state['error_count'] += 1
            raise


    def agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent response with error handling and privacy checks."""
        logger.info(f"Agent state: {state}")
        try:
            ai_response=[self.llm_obj.llm.invoke(state['messages'])]
            return {"messages":ai_response}
            
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            state['error_state'] = True
            state['messages'].append(AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again."))
            return state

    def final_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Final state"""
        logger.info(f"Final state: {state}")
        try:
            return state
        except Exception as e:
            logger.error(f"Error in final state: {str(e)}")
            state['error_state'] = True
            state['messages'].append(AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again."))
            return state
    
