from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    include_metadata: bool = Field(default=True, description="Include document metadata in response")
    max_results: Optional[int] = Field(default=None, ge=1, le=20, description="Maximum number of results to return")

class DocumentMetadata(BaseModel):
    policy_name: str
    page_no: str
    
class SearchResult(BaseModel):
    document_id: str
    content: str
    distance: float
    rerank_score: float
    metadata: DocumentMetadata

class Citation(BaseModel):
    policy_name: str
    page_numbers: List[str]

class QueryResponse(BaseModel):
    query: str
    response: str
    citations: List[Citation]
    search_results: Optional[List[SearchResult]] = None
    from_cache: bool
    processing_time_ms: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    vector_store_status: str
    total_documents: int
    uptime_seconds: float
