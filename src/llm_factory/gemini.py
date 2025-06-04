from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

class GoogleGen:
    def __init__(self, model='gemini-1.5-flash'):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,
            max_output_tokens=2000,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
    
    def __call__(self, messages):
        return self.llm.invoke(messages)