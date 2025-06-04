from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

class GoogleGen:
    def __init__(self, model='gemini-1.5-flash'):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,
            max_output_tokens=2000,
        )
    
    def __call__(self, messages):
        return self.llm.invoke(messages)