from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file in the project root
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(env_path)

class GoogleGen:
    def __init__(self, model='gemini-1.5-flash'):
        # Get API key from environment variables
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,
            max_output_tokens=2000,
            google_api_key=api_key  # Explicitly pass the API key
        )
    
    def __call__(self, messages):
        return self.llm.invoke(messages)