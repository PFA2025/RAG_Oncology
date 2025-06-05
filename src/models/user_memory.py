from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

# Import database configuration
from src.config.database import Base, engine

class UserMemory(Base):
    __tablename__ = "user_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, unique=True)
    name = Column(String, index=True, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self):
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description
        }
        
        # Only include timestamps if they exist
        if hasattr(self, 'created_at') and self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()
            
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            result["updated_at"] = self.updated_at.isoformat()
            
        return result

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)
