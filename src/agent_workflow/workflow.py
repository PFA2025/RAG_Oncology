from langgraph.graph import StateGraph
from agent_workflow.nodes import Nodes
from agent_workflow.state import State
from typing import Dict, Any, Callable, Generator, Optional, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkFlow:
    def __init__(self):
        """Initialize the oncology assistant workflow"""
        try:
            self.nodes = Nodes()
            self.workflow = StateGraph(State)
            
            # Define nodes
            self._setup_nodes()
            
            # Define edges
            self._setup_edges()
            
            # Compile with checkpointing
            memory = MemorySaver()
            self.workflow = self.workflow.compile(checkpointer=memory)
            self.config={'configurable':{'thread_id':'1'}}
                
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {str(e)}")
            raise

    def _setup_nodes(self):
        """Setup all workflow nodes"""
        try:
            self.workflow.add_node('initiate_state', self.nodes.initiate_state)
            self.workflow.add_node('document_retriever', self.nodes.document_retriever)
            self.workflow.add_node('relevance_checker', self.nodes.relevance_checker)
            self.workflow.add_node('prepare_prompt', self.nodes.prepare_prompt)
            self.workflow.add_node('agent', self.nodes.agent)
            self.workflow.add_node('final_state', self.nodes.final_state)
            logger.info("Nodes setup completed")
        except Exception as e:
            logger.error(f"Error setting up nodes: {str(e)}")
            raise

    def _setup_edges(self):
        """Setup all workflow edges"""
        try:
            # Basic flow
            self.workflow.add_edge("__start__", 'initiate_state')
            self.workflow.add_edge('initiate_state', 'document_retriever')
            self.workflow.add_edge('document_retriever', 'relevance_checker')
            self.workflow.add_conditional_edges('relevance_checker', condition_function,{True:'prepare_prompt',False:"final_state"})
            self.workflow.add_edge('agent', 'final_state')
            self.workflow.add_edge('final_state', "__end__")
            logger.info("Edges setup completed")
        except Exception as e:
            logger.error(f"Error setting up edges: {str(e)}")
            raise


    def show_state(self) -> None:
        """Display the current conversation state"""
        try:
            state = self.workflow.get_state(self.config)
            if not state or not state.values.get('messages'):
                logger.warning("No messages in current state")
                return
                
            print("\n=== Conversation History ===")
            for m in state.values['messages']:
                print(f"{type(m).__name__}: {m.content}")
            print("==========================\n")
            
            # Show metadata summary
            if state.values.get('metadata'):
                meta = state.values['metadata']
                print(f"Session ID: {state.values.get('session_id')}")
                print(f"Started: {meta.get('session_start')}")
                print(f"Interactions: {len(meta.get('interactions', []))}")
                print(f"Avg Confidence: {meta.get('avg_confidence', 0.0):.2f}")
        except Exception as e:
            logger.error(f"Error showing state: {str(e)}")

    def return_state_value(self, state_name: str) -> Optional[list]:
        """Return specific state values"""
        try:
            state = self.workflow.get_state(self.config)
            if not state or state_name not in state.values:
                logger.warning(f"State '{state_name}' not found")
                return None
                
            value = state.values[state_name]
            return list(value) if isinstance(value, (list, tuple)) else [value]
        except Exception as e:
            logger.error(f"Error returning state value: {str(e)}")
            return None