from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import relationship

# Import database configuration
from src.config.database import Base, engine

class UserMemory(Base):
    __tablename__ = "user_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=True)
    description = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)
