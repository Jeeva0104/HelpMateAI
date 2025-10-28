from sentence_transformers import CrossEncoder
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from app.models import SearchResult, DocumentMetadata
from app.services.embedding_service import EmbeddingService
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class SearchService:
    """Service for handling search operations including semantic search and re-ranking."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.cross_encoder = CrossEncoder(settings.cross_encoder_model)
        self.main_collection = None
        self.cache_collection = None
        self.logger = logger
        
    def initialize_collections(self):
        """Initialize main and cache collections."""
        self.main_collection, self.cache_collection = self.embedding_service.initialize_collections()
        
    def search_with_cache(self, query: str, n_results: Optional[int] = None) -> Tuple[List[SearchResult], bool]:
        """
        Search with cache-first approach.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Tuple of (search_results, from_cache)
        """
        if n_results is None:
            n_results = settings.search_results_limit
            
        try:
            # Initialize collections if not already done
            if self.cache_collection is None:
                self.initialize_collections()
            
            # 1. Check cache first
            cache_results = self.cache_collection.query(query_texts=query, n_results=1)
            
            from_cache = False
            
            # If cache miss or distance above threshold, search main collection
            if (cache_results["distances"][0] == [] or 
                cache_results["distances"][0][0] > settings.cache_threshold):
                
                self.logger.info("Cache miss - searching main collection")
                search_results = self.semantic_search(query, n_results)
                
                # Store in cache for future use
                self._store_in_cache(query, search_results)
                
            else:
                self.logger.info("Cache hit - retrieving from cache")
                search_results = self._retrieve_from_cache(cache_results)
                from_cache = True
            
            # Re-rank results using cross-encoder
            reranked_results = self.rerank_results(query, search_results)
            
            # Return top results after re-ranking
            top_results = reranked_results[:settings.rerank_top_k]
            
            return top_results, from_cache
            
        except Exception as e:
            self.logger.error(f"Error in search_with_cache: {e}")
            raise
        
    def semantic_search(self, query: str, n_results: int) -> List[SearchResult]:
        """
        Perform semantic search on main collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if self.main_collection is None:
                self.initialize_collections()
            
            # Query the collection
            results = self.main_collection.query(
                query_texts=query, 
                n_results=n_results
            )
            
            # Convert to SearchResult objects
            search_results = []
            for i in range(len(results["ids"][0])):
                metadata = DocumentMetadata(
                    policy_name=results["metadatas"][0][i]["Policy_Name"],
                    page_no=results["metadatas"][0][i]["Page_No."]
                )
                
                search_result = SearchResult(
                    document_id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    distance=results["distances"][0][i],
                    rerank_score=0.0,  # Will be set during re-ranking
                    metadata=metadata
                )
                search_results.append(search_result)
            
            self.logger.info(f"Found {len(search_results)} semantic search results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in semantic_search: {e}")
            raise
        
    def rerank_results(self, query: str, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Re-rank search results using cross-encoder.
        
        Args:
            query: Original query
            search_results: List of search results to re-rank
            
        Returns:
            Re-ranked list of search results
        """
        try:
            if not search_results:
                return search_results
            
            # Create input pairs for cross-encoder
            cross_inputs = [[query, result.content] for result in search_results]
            
            # Get re-ranking scores
            rerank_scores = self.cross_encoder.predict(cross_inputs)
            
            # Update search results with rerank scores
            for i, result in enumerate(search_results):
                result.rerank_score = float(rerank_scores[i])
            
            # Sort by rerank score (descending)
            reranked_results = sorted(
                search_results, 
                key=lambda x: x.rerank_score, 
                reverse=True
            )
            
            self.logger.info(f"Re-ranked {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Error in rerank_results: {e}")
            raise
    
    def _store_in_cache(self, query: str, search_results: List[SearchResult]):
        """Store search results in cache for future retrieval."""
        try:
            # Prepare cache metadata
            cache_metadata = {}
            for i, result in enumerate(search_results):
                cache_metadata[f"ids{i}"] = result.document_id
                cache_metadata[f"documents{i}"] = result.content
                cache_metadata[f"distances{i}"] = str(result.distance)
                cache_metadata[f"metadatas{i}"] = result.metadata.page_no
            
            # Store in cache collection
            self.cache_collection.add(
                documents=[query],
                ids=[query],
                metadatas=[cache_metadata]
            )
            
            self.logger.info("Stored results in cache")
            
        except Exception as e:
            self.logger.error(f"Error storing in cache: {e}")
    
    def _retrieve_from_cache(self, cache_results: Dict[str, Any]) -> List[SearchResult]:
        """Retrieve search results from cache."""
        try:
            cache_metadata = cache_results["metadatas"][0][0]
            
            search_results = []
            
            # Reconstruct search results from cache metadata
            i = 0
            while f"ids{i}" in cache_metadata:
                metadata = DocumentMetadata(
                    policy_name="Principal-Sample-Life-Insurance-Policy",
                    page_no=cache_metadata[f"metadatas{i}"]
                )
                
                search_result = SearchResult(
                    document_id=cache_metadata[f"ids{i}"],
                    content=cache_metadata[f"documents{i}"],
                    distance=float(cache_metadata[f"distances{i}"]),
                    rerank_score=0.0,  # Will be set during re-ranking
                    metadata=metadata
                )
                search_results.append(search_result)
                i += 1
            
            self.logger.info(f"Retrieved {len(search_results)} results from cache")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving from cache: {e}")
            raise
