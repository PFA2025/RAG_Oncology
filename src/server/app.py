from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
import traceback
from datetime import datetime
from sqlalchemy.orm import Session

# Import database configuration and models
from src.config.database import get_db, Base, engine
from src.models.user_memory import UserMemory, init_db
from src.helpers.user_memory_manager import UserMemoryManager

# Initialize the database
init_db()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cancer Agent API")

# Add middleware for request logging
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise

app.middleware('http')(log_requests)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize any required services here

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    source: Optional[str] = None

# User Memory Models
class UserMemoryBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: int

class UserMemoryCreate(UserMemoryBase):
    pass

class UserMemoryUpdate(UserMemoryBase):
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: Optional[int] = None  # Optional for updates

class UserMemoryResponse(UserMemoryBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # Updated from orm_mode for Pydantic v2

@app.get("/")
async def root():
    return {"message": "Cancer Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# User Memory Endpoints
@app.post("/user-memories/", response_model=UserMemoryResponse, status_code=status.HTTP_201_CREATED)
def create_user_memory(user_memory: UserMemoryCreate, db: Session = Depends(get_db)):
    """
    Create a new user memory
    
    Note: Each user can only have one memory entry.
    """
    try:
        return UserMemoryManager.create_memory(
            user_id=user_memory.user_id,
            name=user_memory.name,
            description=user_memory.description
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user-memories/user/{user_id}", response_model=UserMemoryResponse)
def read_user_memory_by_user(user_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific user memory by user ID"""
    memory = UserMemoryManager.get_memory_by_user(user_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"No memory found for user {user_id}")
    return memory

@app.put("/user-memories/user/{user_id}", response_model=UserMemoryResponse)
def update_user_memory_by_user(
    user_id: int, 
    user_memory: UserMemoryUpdate, 
    db: Session = Depends(get_db)
):
    """Update a user memory by user ID"""
    try:
        update_data = user_memory.dict(exclude_unset=True)
        # Remove user_id from update data if it's None to avoid overwriting
        if 'user_id' in update_data and update_data['user_id'] is None:
            del update_data['user_id']
            
        updated = UserMemoryManager.update_memory(
            user_id=user_id,
            name=update_data.get('name'),
            description=update_data.get('description')
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"No memory found for user {user_id}")
        return updated
    except Exception as e:
        logger.error(f"Error updating user memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/user-memories/user/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_memory_by_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user memory by user ID"""
    if not UserMemoryManager.delete_memory(user_id):
        raise HTTPException(status_code=404, detail=f"No memory found for user {user_id}")
    return None

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Simple chat endpoint that echoes the message back"""
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