from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging

# Import database configuration
from src.config.database import SessionLocal, get_db

@contextmanager
def get_db_session():
    """Provide a transactional scope around a series of operations."""
    db = next(get_db())
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

class UserMemoryManager:
    """
    A class to manage user memories directly in the database.
    """
    
    @staticmethod
    def create_memory(user_id: int, name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new user memory.
        
        Args:
            user_id: The ID of the user this memory belongs to
            name: Optional name for the memory
            description: Optional description of the memory
            
        Returns:
            dict: The created user memory
        """
        from src.models.user_memory import UserMemory
        
        try:
            with get_db_session() as session:
                # Check if user already has a memory
                existing_memory = session.query(UserMemory).filter(UserMemory.user_id == user_id).first()
                if existing_memory:
                    raise ValueError(f"User {user_id} already has a memory entry")
                    
                memory = UserMemory(
                    user_id=user_id,
                    name=name,
                    description=description
                )
                session.add(memory)
                session.commit()
                session.refresh(memory)
                return memory.to_dict()
        except Exception as e:
            # Log error and re-raise
            logging.error(f"Error creating memory: {str(e)}")
            raise

    @staticmethod
    def get_memory_by_id(memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user memory by its internal ID.
        
        Args:
            memory_id: The internal ID of the memory
            
        Returns:
            Optional[dict]: The user memory if found, None otherwise
        """
        from src.models.user_memory import UserMemory
        
        try:
            with get_db_session() as session:
                memory = session.query(UserMemory).filter(UserMemory.id == memory_id).first()
                return memory.to_dict() if memory else None
        except Exception as e:
            logging.error(f"Error getting memory by ID {memory_id}: {str(e)}")
            raise
    
    @staticmethod
    def get_memory_by_user(user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user memory by user ID.
        
        Args:
            user_id: The ID of the user whose memory to retrieve
            
        Returns:
            Optional[dict]: The user memory if found, None otherwise
        """
        from src.models.user_memory import UserMemory
        
        try:
            with get_db_session() as session:
                memory = session.query(UserMemory).filter(UserMemory.user_id == user_id).first()
                return memory.to_dict() if memory else None
        except Exception as e:
            logging.error(f"Error getting memory {memory_id}: {str(e)}")
            raise
    
    @staticmethod
    def update_memory(user_id: int, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Update a user's memory by user ID.
        
        Args:
            user_id: The ID of the user whose memory to update
            name: New name for the memory (optional)
            description: New description for the memory (optional)
            
        Returns:
            dict: The updated user memory if found, None otherwise
        """
        from src.models.user_memory import UserMemory
        
        try:
            with get_db_session() as session:
                memory = session.query(UserMemory).filter(UserMemory.user_id == user_id).first()
                if not memory:
                    return None
                    
                if name is not None:
                    memory.name = name
                if description is not None:
                    memory.description = description
                    
                session.commit()
                session.refresh(memory)
                return memory.to_dict()
        except Exception as e:
            logging.error(f"Error updating memory {memory_id}: {str(e)}")
            raise
    
    @staticmethod
    def delete_memory(user_id: int) -> bool:
        """
        Delete a user memory by user ID.
        
        Args:
            user_id: The ID of the user whose memory to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        from src.models.user_memory import UserMemory
        
        try:
            with get_db_session() as session:
                memory = session.query(UserMemory).filter(UserMemory.user_id == user_id).first()
                if not memory:
                    return False
                    
                session.delete(memory)
                session.commit()
                return True
        except Exception as e:
            logging.error(f"Error deleting memory {memory_id}: {str(e)}")
            raise
