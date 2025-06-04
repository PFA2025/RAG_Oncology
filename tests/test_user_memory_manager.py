import pytest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.helpers.user_memory_manager import UserMemoryManager
from src.models.user_memory import UserMemory
from src.config.database import SessionLocal

@pytest.fixture(scope="module")
def db_session():
    """Create a clean database session for tests"""
    db = SessionLocal()
    try:
        # Clean up any existing data
        db.query(UserMemory).delete()
        db.commit()
        yield db
    finally:
        db.close()

def test_create_memory(db_session):
    """Test creating a new memory"""
    memory_data = {"name": "Test User", "description": "Test description"}
    result = UserMemoryManager.create_memory(**memory_data)
    
    assert "id" in result
    assert result["name"] == "Test User"
    assert result["description"] == "Test description"

def test_get_memory(db_session):
    """Test retrieving a memory"""
    # Create a memory first
    memory_data = {"name": "Test User", "description": "Test description"}
    created = UserMemoryManager.create_memory(**memory_data)
    
    # Retrieve it
    result = UserMemoryManager.get_memory(created["id"])
    
    assert result["id"] == created["id"]
    assert result["name"] == "Test User"

def test_update_memory(db_session):
    """Test updating a memory"""
    # Create a memory first
    created = UserMemoryManager.create_memory(name="Original", description="Original")
    
    # Update it
    updated = UserMemoryManager.update_memory(
        memory_id=created["id"],
        name="Updated",
        description="Updated"
    )
    
    assert updated["name"] == "Updated"
    assert updated["description"] == "Updated"

def test_delete_memory(db_session):
    """Test deleting a memory"""
    # Create a memory first
    created = UserMemoryManager.create_memory(name="To Delete", description="Test")
    
    # Delete it
    delete_result = UserMemoryManager.delete_memory(created["id"])
    assert delete_result is True
    
    # Verify it's gone
    result = UserMemoryManager.get_memory(created["id"])
    assert result is None
