from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from agent_workflow.workflow import WorkFlow
from langchain_core.messages import HumanMessage, AIMessage
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cancer Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow = WorkFlow()

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    source: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Cancer Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Replace the chat endpoint with this:
# Replace your chat endpoint with this:
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        work_flow = WorkFlow()
        response = work_flow(message.message)
        ai_response = response.get('messages', [])[-1].content
        logger.info(f"AI response: {ai_response}")
        
        # # Get the AI response
        # ai_messages = [m for m in state.get('messages', []) if isinstance(m, AIMessage)]
        # if not ai_messages:
        #     raise HTTPException(status_code=500, detail="No response generated")
        
        # last_ai_message = ai_messages[-1].content
        
        # # Get confidence from relevance checks
        # relevance_checks = state.get('metadata', {}).get('relevance_checks', [])
        # confidence = relevance_checks[-1]['confidence'] if relevance_checks else 0.0
        
        # # Get source from answer result
        # source = state.get('answer_result', {}).get('source', 'unknown')
        
        # return ChatResponse(
        #     response=last_ai_message,
        #     confidence=confidence,
        #     source=source
        # )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))