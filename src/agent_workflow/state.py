from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from datetime import datetime
import uuid
from enum import Enum
import logging
# Add this to the imports:
from enum import Enum

# Add this enum above the StateMetadata class:
class SourceType(str, Enum):
    VERIFIED = "verified_answer"
    GENERATED = "llm_generated"
    ENHANCED = "llm_enhanced"
    ERROR = "error"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrivacyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SourceType(str, Enum):
    VERIFIED = "verified_answer"
    GENERATED = "llm_generated"
    ENHANCED = "llm_enhanced"
    ERROR = "error"

class Interaction(TypedDict):
    timestamp: str
    query: str
    answer: str
    source: SourceType
    confidence: float
    context_used: bool

class StateMetadata(TypedDict):
    session_start: str
    session_end: Optional[str]
    version: str
    interactions: List[Interaction]
    error_count: int
    avg_confidence: Optional[float]
    privacy_level: PrivacyLevel

class AgentState(TypedDict):
    # Core conversation state
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Session tracking
    session_id: str
    current_query: Optional[str]
    last_active: str
    
    # Context management
    retrieved_context: Optional[List[Dict[str, Any]]]
    current_response: Optional[Dict[str, Any]]
    
    # Workflow control
    should_continue: bool
    error_state: bool
    
    # Metadata storage
    metadata: StateMetadata

def new_state() -> AgentState:
    """Initialize a new conversation state with default values"""
    session_id = f"oncosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    return {
        "messages": [],
        "session_id": session_id,
        "current_query": None,
        "last_active": datetime.now().isoformat(),
        "retrieved_context": None,
        "current_response": None,
        "should_continue": True,
        "error_state": False,
        "metadata": {
            "session_start": datetime.now().isoformat(),
            "session_end": None,
            "version": "1.1.0",
            "interactions": [],
            "error_count": 0,
            "avg_confidence": None,
            "privacy_level": PrivacyLevel.HIGH
        }
    }

def update_state_metadata(
    state: AgentState,
    query: str,
    answer: str,
    source: SourceType,
    confidence: float,
    context_used: bool
) -> AgentState:
    """Update the state metadata with a new interaction"""
    try:
        new_interaction: Interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "source": source,
            "confidence": confidence,
            "context_used": context_used
        }
        
        state["metadata"]["interactions"].append(new_interaction)
        
        # Update average confidence
        interactions = state["metadata"]["interactions"]
        if interactions:
            total = sum(i["confidence"] for i in interactions)
            state["metadata"]["avg_confidence"] = total / len(interactions)
        
        return state
    except Exception as e:
        logger.error(f"Error updating state metadata: {str(e)}")
        state["metadata"]["error_count"] += 1
        return state

def validate_state(state: Dict[str, Any]) -> bool:
    """Validate the state structure"""
    required_keys = {
        "messages", "session_id", "metadata",
        "should_continue", "error_state"
    }
    
    if not all(key in state for key in required_keys):
        return False
        
    if not isinstance(state["metadata"], dict):
        return False
        
    return True

def get_active_context(state: AgentState) -> List[Dict[str, Any]]:
    """Get the currently active context from the state"""
    return state.get("retrieved_context", [])

def get_last_response(state: AgentState) -> Optional[Dict[str, Any]]:
    """Get the last generated response"""
    return state.get("current_response")

def finalize_state(state: AgentState) -> AgentState:
    """Prepare the state for session completion"""
    try:
        state["metadata"]["session_end"] = datetime.now().isoformat()
        state["should_continue"] = False
        
        # Calculate final metrics
        interactions = state["metadata"]["interactions"]
        if interactions:
            state["metadata"]["avg_confidence"] = (
                sum(i["confidence"] for i in interactions) / len(interactions)
            )
        
        return state
    except Exception as e:
        logger.error(f"Error finalizing state: {str(e)}")
        state["error_state"] = True
        return state