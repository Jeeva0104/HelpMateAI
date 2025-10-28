from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import time
from datetime import datetime
from typing import Optional
import os

from app.models import QueryRequest, QueryResponse, HealthResponse, SearchResult, SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.generation_service import GenerationService
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global service instances
embedding_service: Optional[EmbeddingService] = None
search_service: Optional[SearchService] = None
generation_service: Optional[GenerationService] = None

# Application startup time for uptime calculation
app_start_time = time.time()

app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation Pipeline for Insurance Documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global embedding_service, search_service, generation_service
    
    logger.info("Starting RAG Pipeline API...")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        search_service = SearchService(embedding_service)
        generation_service = GenerationService()
        
        # Initialize collections
        search_service.initialize_collections()
        
        logger.info("Services initialized successfully")
        
        # Verify vector store is ready
        if not embedding_service.vector_store_exists():
            logger.warning("Vector store is empty. Make sure the startup script has run successfully.")
        else:
            stats = embedding_service.get_collection_stats()
            logger.info(f"Vector store ready with {stats.get('main_collection_count', 0)} documents")
            
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def get_embedding_service() -> EmbeddingService:
    """Dependency to get embedding service."""
    if embedding_service is None:
        raise HTTPException(status_code=500, detail="Embedding service not initialized")
    return embedding_service

def get_search_service() -> SearchService:
    """Dependency to get search service."""
    if search_service is None:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    return search_service

def get_generation_service() -> GenerationService:
    """Dependency to get generation service."""
    if generation_service is None:
        raise HTTPException(status_code=500, detail="Generation service not initialized")
    return generation_service

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    search_svc: SearchService = Depends(get_search_service),
    generation_svc: GenerationService = Depends(get_generation_service)
):
    """
    Complete RAG pipeline endpoint: Search + Generation
    
    This endpoint performs the following steps:
    1. Semantic search with caching
    2. Re-ranking using cross-encoder
    3. Response generation using LLM
    4. Citation extraction
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Search with cache (includes re-ranking)
        search_results, from_cache = search_svc.search_with_cache(
            query=request.query,
            n_results=request.max_results or settings.search_results_limit
        )
        
        if not search_results:
            raise HTTPException(
                status_code=404, 
                detail="No relevant documents found for your query"
            )
        
        # Step 2: Generate response using LLM
        full_response = generation_svc.generate_response(
            query=request.query,
            search_results=search_results
        )
        
        # Step 3: Parse response and extract citations
        parsed_response = generation_svc.parse_response_and_citations(full_response)
        citations = generation_svc.extract_citations(search_results)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response = QueryResponse(
            query=request.query,
            response=parsed_response["response"],
            citations=citations,
            search_results=search_results if request.include_metadata else None,
            from_cache=from_cache,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Query processed successfully in {processing_time:.2f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while processing query: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    search_svc: SearchService = Depends(get_search_service)
):
    """
    Search endpoint that returns top 3 results from the search layer only.
    
    This endpoint performs:
    1. Semantic search with caching
    2. Re-ranking using cross-encoder
    3. Returns top 3 search results without generation
    
    Example usage for "How do I file a death benefit claim?"
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing search query: {request.query}")
        
        # Perform search with cache (includes semantic search and re-ranking)
        search_results, from_cache = search_svc.search_with_cache(
            query=request.query,
            n_results=settings.search_results_limit
        )
        
        if not search_results:
            raise HTTPException(
                status_code=404, 
                detail="No relevant documents found for your query"
            )
        
        # Return only top 3 results
        top_3_results = search_results[:3]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response = SearchResponse(
            query=request.query,
            search_results=top_3_results,
            from_cache=from_cache,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Search completed successfully in {processing_time:.2f}ms, returned {len(top_3_results)} results")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while processing search: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check(
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Health check endpoint to verify system status.
    
    Returns information about:
    - Service status
    - Vector store status
    - Total documents
    - Uptime
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        # Get vector store statistics
        stats = embedding_svc.get_collection_stats()
        
        # Determine vector store status
        if 'error' in stats:
            vector_store_status = f"Error: {stats['error']}"
            total_documents = 0
        else:
            total_documents = stats.get('main_collection_count', 0)
            if total_documents > 0:
                vector_store_status = "Ready"
            else:
                vector_store_status = "Empty"
        
        response = HealthResponse(
            status="Healthy",
            timestamp=datetime.now(),
            vector_store_status=vector_store_status,
            total_documents=total_documents,
            uptime_seconds=uptime_seconds
        )
        
        logger.info(f"Health check: {response.status}, Vector store: {vector_store_status}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/chat")
async def chat_interface():
    """Serve the chatbot interface."""
    return FileResponse("app/static/chatbot.html")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation Pipeline for Insurance Documents",
        "endpoints": {
            "query": "/query - POST - Submit queries to the RAG pipeline",
            "search": "/search - POST - Get top 3 search results from the retrieval layer",
            "chat": "/chat - GET - Access the chatbot interface",
            "health": "/health - GET - Check system health",
            "docs": "/docs - GET - Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
