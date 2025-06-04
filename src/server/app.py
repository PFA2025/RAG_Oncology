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

class UserMemoryCreate(UserMemoryBase):
    pass

class UserMemoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class UserMemoryResponse(UserMemoryBase):
    id: int

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
    """Create a new user memory"""
    return UserMemoryManager.create_memory(
        name=user_memory.name,
        description=user_memory.description
    )

@app.get("/user-memories/{item_id}", response_model=UserMemoryResponse)
def read_user_memory(item_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific user memory by id"""
    memory = UserMemoryManager.get_memory(item_id)
    if not memory:
        raise HTTPException(status_code=404, detail="User memory not found")
    return memory

@app.put("/user-memories/{item_id}", response_model=UserMemoryResponse)
def update_user_memory(item_id: int, user_memory: UserMemoryUpdate, db: Session = Depends(get_db)):
    """Update a user memory"""
    updated = UserMemoryManager.update_memory(
        memory_id=item_id,
        name=user_memory.name,
        description=user_memory.description
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User memory not found")
    return updated

@app.delete("/user-memories/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_memory(item_id: int, db: Session = Depends(get_db)):
    """Delete a user memory"""
    if not UserMemoryManager.delete_memory(item_id):
        raise HTTPException(status_code=404, detail="User memory not found")
    return None

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Simple chat endpoint that echoes the message back"""
    try:
        return ChatResponse(
            response=f"Echo: {message.message}",
            confidence=1.0,
            source="echo_service"
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))