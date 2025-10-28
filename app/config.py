from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_key: str
    base_url: str
    model: str = "gemini-2.5-flash"
    
    # ChromaDB Configuration
    chroma_db_path: str = "/app/chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Collections
    vector_store_collection: str = "Principal_Life_Insurance"
    cache_collection: str = "Principal_Insurance_Cache"
    
    # Search Parameters
    search_results_limit: int = 10
    rerank_top_k: int = 3
    cache_threshold: float = 0.2
    
    # Application
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
