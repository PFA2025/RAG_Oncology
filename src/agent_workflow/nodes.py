import sys
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.schema import Document

from src.llm_factory.gemini import GoogleGen
from src.helpers.relevance_checker import *
from src.helpers.document_retriever import *
from src.config.logs import get_logger

# Import database configuration and models
from src.config.database import get_db, Base, engine
from src.models.user_memory import UserMemory, init_db
from src.helpers.user_memory_manager import UserMemoryManager

# Initialize the database
init_db()

import time
import os
from pathlib import Path
import json
import uuid
from typing import Dict, Any

# Initialize logger
logger = get_logger(__name__)

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
        logger.info(f"Initializing conversation state with patient_id: {state.get('patient_id')}")
        
        # Initialize patient info in state
        state['patient_name'] = ""
        state['patient_description'] = ""
        
        # If patient_id is provided, fetch the patient info
        patient_id = state.get('patient_id', 0)
        if patient_id and int(patient_id) > 0:
            try:
                patient_info = UserMemoryManager.get_memory_by_user(patient_id)
                if patient_info:
                    state['patient_name'] = patient_info.get('name', '')
                    state['patient_description'] = patient_info.get('description', '')
                    logger.info(f"Patient info loaded: {state['patient_name']}")
                else:
                    logger.warning(f"No patient found with ID: {patient_id}")
            except Exception as e:
                logger.error(f"Error fetching patient info: {str(e)}")
        
        return state
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
            search_results = search_qa(query)
            
            if not search_results:
                logger.warning("No search results found for query")
                state["search_results"] = []
                state["messages"].append(
                    AIMessage(content="I couldn't find any relevant information in my knowledge base. Could you please rephrase your question or provide more details?")
                )
            else:
                state["search_results"] = search_results
            
            logger.info(f"Search results: {state['search_results']}")
            return state
        
        except Exception as e:
            logger.error(f"Error in document retriever: {str(e)}")
            state["error_state"] = True
            state["messages"].append(
                AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again.")
            )
            
            logger.info(f"Error in document retriever: {str(e)}")
            return state
            
    def relevance_checker(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check relevance of search results"""
        logger.info(f"Running relevance checker")
        try:
            # Skip if no search results
            if not state.get("search_results"):
                logger.warning("No search results to check for relevance")
                return state
                
            # Check relevance for each result
            for result in state["search_results"]:
                result["is_relevant"] = check_relevance(state["user_input"], result)
            
            # Filter out irrelevant results
            state["search_results"] = [
                result for result in state["search_results"] 
                if result.get("is_relevant", False)
            ]
            
            if not state["search_results"]:
                logger.info("No relevant results found after relevance check")
                state["messages"].append(
                    AIMessage(content="I couldn't find any relevant information for your query. Could you please provide more details or rephrase your question?")
                )
                
            logger.info(f"Search results: {state['search_results']}")
            return state
            
        except Exception as e:
            logger.error(f"Error in relevance checker: {str(e)}")
            state["error_state"] = True
            state["messages"].append(
                AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again.")
            )
            
            logger.info(f"Error in relevance checker: {str(e)}")
            return state
            
    def prepare_prompt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the system prompt with guidelines and integrate dynamic content.
        
        Args:
            state (Dict[str, Any]): Current state containing user input and search results
            
        Returns:
            Dict[str, Any]: Updated state with system and user messages
        """
        logger.info("Preparing system prompt with dynamic content")
        try:
            # Read template from file
            with open(os.path.abspath(os.path.join(current_dir, "..", "prompts/guidelines.txt")), "r") as file:
                template = file.read()

            # Format sources
            sources = []
            for i, res in enumerate(state.get('search_results', []), 1):
                if 'answer' in res and 'question' in res:
                    sources.append(f"Source {i}:\nQ: {res['question']}\nA: {res['answer']}")
            
            sources_text = "\n\n".join(sources) if sources else "No relevant sources found."
            
            # Log patient info for debugging
            logger.info(f"Using patient info - Name: {state.get('patient_name', 'N/A')}, Description: {state.get('patient_description', 'N/A')}")
            
            # Format the template with dynamic content using str.format()
            system_content = template.format(
                sources=sources_text,
                question=state['user_input'],
                patient_name=state.get('patient_name', ''),
                patient_description=state.get('patient_description', '')
            )
            
            # Create system message with formatted guidelines
            system_message = SystemMessage(content=system_content)
            
            # User message contains just the query
            user_message = HumanMessage(content=state['user_input'])
            
            logger.debug(f"System message prepared with {len(sources)} sources")
            
            # Add messages to state
            messages = state.get('messages', [])
            messages.extend([system_message, user_message])
            
            logger.info(f"Messages: {messages}")
            return {'messages': messages}
            
        except Exception as e:
            logger.error(f"Error in prepare_prompt: {str(e)}")
            state['error_state'] = True
            state['messages'].append(
                AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again.")
            )
            return state


    def agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent response with error handling and privacy checks."""
        logger.info(f"Running the agent state")
        try:
            ai_response=[self.llm_obj.llm.invoke(state['messages'])]
            logger.info(f"AI Response: {ai_response}")
            return {"messages":ai_response}
            
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            state['error_state'] = True
            state['messages'].append(AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again."))
            return state

    def final_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final state processing. Adds source references to the AI response.
        
        Args:
            state: The current state dictionary containing messages and search results
            
        Returns:
            Updated state with sources added to the AI response
        """
        logger.info("Processing final state and adding sources to response")
        try:
            # Get the last AI message
            if 'messages' in state and state['messages']:
                last_message = state['messages'][-1]
                
                # Format sources
                sources = []
                for i, res in enumerate(state.get('search_results', []), 1):
                    if 'answer' in res and 'question' in res:
                        sources.append(f"Source {i}:\nQ: {res['question']}\nA: {res['answer']}")
                
                if sources:
                    sources_text = "\n---\n**Sources:**\n" + "\n".join(sources)
                    # Append sources to the last message
                    state['messages'][-1] = AIMessage(
                        content=last_message.content + sources_text
                    )
                    logger.debug(f"Added {len(sources)} sources to the response")
                else:
                    logger.debug("No valid sources found to add to the response")
            
            return state
        except Exception as e:
            logger.error(f"Error in final state: {str(e)}")
            state['error_state'] = True
            state['messages'].append(AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again."))
            return state
    