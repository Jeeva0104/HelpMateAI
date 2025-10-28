"""Custom exceptions for the RAG Pipeline application."""

class RAGPipelineException(Exception):
    """Base exception class for RAG Pipeline."""
    pass

class VectorStoreException(RAGPipelineException):
    """Exception raised for vector store related errors."""
    pass

class EmbeddingException(RAGPipelineException):
    """Exception raised for embedding related errors."""
    pass

class SearchException(RAGPipelineException):
    """Exception raised for search related errors."""
    pass

class GenerationException(RAGPipelineException):
    """Exception raised for response generation errors."""
    pass

class DocumentProcessingException(RAGPipelineException):
    """Exception raised for document processing errors."""
    pass

class ConfigurationException(RAGPipelineException):
    """Exception raised for configuration related errors."""
    pass
