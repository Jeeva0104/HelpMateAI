#!/usr/bin/env python3
"""
Startup script to build vector store from documents.
This serves as the Embedding Layer - processes documents and creates embeddings.
"""

import os
import sys
import logging
from pathlib import Path

# Add app to Python path
sys.path.insert(0, '/app')

from app.utils.pdf_processor import process_documents
from app.services.embedding_service import EmbeddingService
from app.config import settings
from app.utils.logger import get_logger

def setup_logging():
    """Set up logging for the startup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """
    Main function to build vector store from documents.
    
    This function:
    1. Checks if vector store already exists
    2. Processes PDF documents
    3. Creates embeddings and stores in ChromaDB
    4. Validates the vector store
    """
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("RAG PIPELINE - EMBEDDING LAYER (Startup Script)")
    logger.info("Building Vector Store from Documents")
    logger.info("=" * 60)
    
    try:
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        # Check if vector store already exists and is populated
        if embedding_service.vector_store_exists():
            stats = embedding_service.get_collection_stats()
            doc_count = stats.get('main_collection_count', 0)
            logger.info(f"Vector store already exists with {doc_count} documents")
            logger.info("Skipping document processing...")
            return True
        
        # Define document path
        pdf_path = "/app/data/Principal-Sample-Life-Insurance-Policy.pdf"
        
        # Check if document exists
        if not os.path.exists(pdf_path):
            logger.error(f"Document not found: {pdf_path}")
            logger.error("Please ensure the PDF document is mounted in the /app/data directory")
            return False
        
        logger.info(f"Processing document: {pdf_path}")
        
        # Process documents
        logger.info("Extracting text from PDF...")
        documents_df = process_documents(pdf_path)
        
        if documents_df.empty:
            logger.error("No documents were processed successfully")
            return False
        
        logger.info(f"Successfully processed {len(documents_df)} document pages")
        
        # Create embeddings and store in vector database
        logger.info("Creating embeddings and storing in ChromaDB...")
        success = embedding_service.embed_documents(documents_df)
        
        if not success:
            logger.error("Failed to embed documents")
            return False
        
        # Validate vector store
        logger.info("Validating vector store...")
        stats = embedding_service.get_collection_stats()
        
        if 'error' in stats:
            logger.error(f"Vector store validation failed: {stats['error']}")
            return False
        
        doc_count = stats.get('main_collection_count', 0)
        cache_count = stats.get('cache_collection_count', 0)
        
        logger.info("=" * 60)
        logger.info("VECTOR STORE BUILD COMPLETED SUCCESSFULLY")
        logger.info(f"📚 Documents processed: {doc_count}")
        logger.info(f"🔍 Cache entries: {cache_count}")
        logger.info(f"🤖 Embedding model: {stats.get('embedding_model', 'N/A')}")
        logger.info(f"💾 Storage path: {settings.chroma_db_path}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error building vector store: {e}")
        logger.error("Vector store build FAILED")
        return False

def health_check():
    """Simple health check for container startup verification."""
    try:
        from app.services.embedding_service import EmbeddingService
        service = EmbeddingService()
        stats = service.get_collection_stats()
        
        if 'error' in stats:
            print(f"Health check FAILED: {stats['error']}")
            sys.exit(1)
        
        doc_count = stats.get('main_collection_count', 0)
        if doc_count > 0:
            print(f"Health check PASSED: {doc_count} documents in vector store")
            sys.exit(0)
        else:
            print("Health check FAILED: Vector store is empty")
            sys.exit(1)
            
    except Exception as e:
        print(f"Health check FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if this is a health check call
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        health_check()
    else:
        # Build vector store
        success = main()
        if not success:
            sys.exit(1)
        
        print("\n🚀 Vector store build completed! Starting FastAPI application...")
