import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    """Service for handling document embeddings and ChromaDB operations."""
    
    def __init__(self):
        self.logger = logger
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.main_collection = None
        self.cache_collection = None
        
    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection with embedding function."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Retrieved/created collection: {collection_name}")
            return collection
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            raise
    
    def initialize_collections(self):
        """Initialize main and cache collections."""
        self.main_collection = self.get_or_create_collection(
            settings.vector_store_collection
        )
        self.cache_collection = self.get_or_create_collection(
            settings.cache_collection
        )
        return self.main_collection, self.cache_collection
    
    def embed_documents(self, documents_df: pd.DataFrame) -> bool:
        """
        Embed documents and store in ChromaDB.
        
        Args:
            documents_df: DataFrame with document data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.main_collection is None:
                self.main_collection = self.get_or_create_collection(
                    settings.vector_store_collection
                )
            
            # Check if documents already exist
            existing_count = self.main_collection.count()
            if existing_count > 0:
                self.logger.info(f"Collection already contains {existing_count} documents")
                return True
            
            # Prepare data for embedding
            documents_list = documents_df["Page_Text"].tolist()
            metadata_list = documents_df["Metadata"].tolist()
            ids_list = [str(i) for i in range(len(documents_list))]
            
            # Add documents to collection
            self.main_collection.add(
                documents=documents_list,
                ids=ids_list,
                metadatas=metadata_list,
            )
            
            self.logger.info(f"Successfully embedded {len(documents_list)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error embedding documents: {e}")
            return False
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Use the same embedding function as the collection
            embedding = self.embedding_function([query])[0]
            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding query: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collections."""
        try:
            stats = {}
            
            if self.main_collection is None:
                self.main_collection = self.get_or_create_collection(
                    settings.vector_store_collection
                )
            
            if self.cache_collection is None:
                self.cache_collection = self.get_or_create_collection(
                    settings.cache_collection
                )
            
            stats['main_collection_count'] = self.main_collection.count()
            stats['cache_collection_count'] = self.cache_collection.count()
            stats['embedding_model'] = settings.embedding_model
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def vector_store_exists(self) -> bool:
        """Check if vector store is already populated."""
        try:
            if self.main_collection is None:
                self.main_collection = self.get_or_create_collection(
                    settings.vector_store_collection
                )
            
            count = self.main_collection.count()
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Error checking vector store existence: {e}")
            return False
