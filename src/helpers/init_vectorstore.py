from document_retriever import create_vectorstore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing vector store...")
    vector_store = create_vectorstore()
    if vector_store:
        logger.info("Vector store initialized successfully")
    else:
        logger.error("Failed to initialize vector store")

if __name__ == "__main__":
    main() 