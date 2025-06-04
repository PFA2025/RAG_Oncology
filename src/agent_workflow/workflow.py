from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from agent_workflow.nodes import Nodes
from agent_workflow.state import AgentState, new_state, update_state_metadata, finalize_state, SourceType
from typing import Dict, Any, Callable, Generator, Optional, List
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
            self.workflow = StateGraph(AgentState)
            
            # Define nodes
            self._setup_nodes()
            
            # Define edges
            self._setup_edges()
            
            # Add conditional edges
            self._setup_conditional_edges()
            
            # Compile with checkpointing
            memory = InMemorySaver()
            self.workflow = self.workflow.compile(
                checkpointer=memory,
                interrupt_before=["agent"],
                interrupt_after=["final_state"]
            )
            
            # Workflow configuration
            self.config = {
                'configurable': {
                    'thread_id': str(datetime.now().timestamp()),
                    'session_start': datetime.now().isoformat()
                }
            }
            
            logger.info("Workflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {str(e)}")
            raise

    def _setup_nodes(self):
        """Setup all workflow nodes"""
        try:
            self.workflow.add_node('initiate_state', self.nodes.initiate_state)
            self.workflow.add_node('prepare_prompt', self.nodes.prepare_prompt)
            self.workflow.add_node('user', self.nodes.user)
            self.workflow.add_node('agent', self.nodes.agent)
            self.workflow.add_node('final_state', self._final_state_wrapper)
            self.workflow.add_node('handle_error', self._error_handler)
            logger.info("Nodes setup completed")
        except Exception as e:
            logger.error(f"Error setting up nodes: {str(e)}")
            raise

    def _setup_edges(self):
        """Setup all workflow edges"""
        try:
            # Basic flow
            self.workflow.add_edge("__start__", 'initiate_state')
            self.workflow.add_edge('initiate_state', 'prepare_prompt')
            self.workflow.add_edge('prepare_prompt', 'user')
            self.workflow.add_edge('agent', 'user')
            self.workflow.add_edge('final_state', "__end__")
            self.workflow.add_edge('handle_error', 'user')
            logger.info("Edges setup completed")
        except Exception as e:
            logger.error(f"Error setting up edges: {str(e)}")
            raise

    def _setup_conditional_edges(self):
        """Setup conditional edges for workflow control"""
        try:
            self.workflow.add_conditional_edges(
                'user',
                self.check_user_input,
                {
                    True: 'agent',
                    False: 'final_state'
                }
            )
            logger.info("Conditional edges setup completed")
        except Exception as e:
            logger.error(f"Error setting up conditional edges: {str(e)}")
            raise

    def check_user_input(self, state: AgentState) -> bool:
        """Check if the conversation should continue"""
        try:
            if not state.get('messages'):
                return False
                
            last_message = state['messages'][-1]
            if not isinstance(last_message, BaseMessage):
                return False
                
            return last_message.content.lower() != 'exit'
        except Exception as e:
            logger.error(f"Error checking user input: {str(e)}")
            return False

    def _error_handler(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        try:
            logger.error(f"Error encountered in state: {state}")
            # Update error count in metadata
            state['metadata']['error_count'] += 1
            
            # Prepare error message
            error_message = "I encountered a problem. Please rephrase your question."
            if state['metadata']['error_count'] >= self.nodes.max_retries:
                error_message = "I'm having technical difficulties. Please try again later."
                state['should_continue'] = False
            
            # Create response
            return {
                **state,
                'messages': [AIMessage(content=error_message)],
                'current_response': {
                    'content': error_message,
                    'source': SourceType.ERROR,
                    'confidence': 0.0
                }
            }
        except Exception as e:
            logger.critical(f"Critical error in error handler: {str(e)}")
            return {
                'messages': [AIMessage(content="System error. Ending conversation.")],
                'should_continue': False,
                'error_state': True
            }

    def _final_state_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for final state processing"""
        try:
            # Update state with final response metadata if exists
            if state.get('current_response'):
                response = state['current_response']
                update_state_metadata(
                    state,
                    query=state.get('current_query', "Final state"),
                    answer=response.get('content', "Conversation completed"),
                    source=response.get('source', SourceType.VERIFIED),
                    confidence=response.get('confidence', 1.0),
                    context_used=bool(state.get('retrieved_context'))
                )
            
            # Finalize state
            return finalize_state(state)
        except Exception as e:
            logger.error(f"Error in final state processing: {str(e)}")
            state['error_state'] = True
            return state

    def __call__(self) -> AgentState:
        """Execute the workflow with initial state"""
        try:
            logger.info("Starting workflow execution")
            # Initialize state using the standardized function
            initial_state = new_state()
            # Add system prompt
            initial_state = self.nodes.prepare_prompt(initial_state)
            logger.info("Workflow execution completed")
            return initial_state
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            return {
                'messages': [AIMessage(content=f"System error: {str(e)}")],
                'should_continue': False,
                'error_state': True,
                'metadata': {
                    'session_end': datetime.now().isoformat(),
                    'error_count': 1
                }
            }

    def process_message(self, state: AgentState, message: str) -> AgentState:
        """Process a single message through the workflow"""
        try:
            logger.info("Processing message through workflow")
            # Add user message to state
            state['messages'].append(HumanMessage(content=message))
            
            # Process user node
            user_result = self.nodes.user(state)
            state.update(user_result)
            
            # Check for error state
            if state.get('error_state', False):
                return state
                
            # Process agent node
            agent_result = self.nodes.agent(state)
            state.update(agent_result)
            
            return state
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            state['messages'].append(AIMessage(content="I encountered an error. Please try again."))
            state['error_state'] = True
            return state

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