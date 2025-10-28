from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from app.models import SearchResult, Citation, DocumentMetadata
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class GenerationService:
    """Service for generating responses using LLM based on search results."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model,
            temperature=0,
            openai_api_key=settings.api_key,
            openai_api_base=settings.base_url,
        )
        self.logger = logger
        
    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate response using LLM with context from search results.
        
        Args:
            query: User query
            search_results: List of relevant search results
            
        Returns:
            Generated response string
        """
        try:
            # Format context from search results
            context = self.format_context(search_results)
            
            # Create messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents."
                },
                {
                    "role": "user",
                    "content": f"""You are a helpful assistant in the insurance domain who can effectively answer user queries about insurance policies and documents.
                    
You have a question asked by the user: '{query}'

You have some search results from a corpus of insurance documents below. These search results are essentially pages from insurance documents that may be relevant to the user query.

Search Results:
{context}

Use the search results to answer the query '{query}'. Frame an informative answer and also provide relevant policy names and page numbers as citations.

Follow these guidelines:
1. Try to provide relevant/accurate numbers if available.
2. Only use information that is relevant to the query.
3. If the document text has tables with relevant information, please reformat the table and return the final information in a tabular format.
4. Use the metadata to retrieve and cite the policy name(s) and page number(s) as citations.
5. If you can't provide the complete answer, provide any information that will help the user search specific sections in the relevant cited documents.
6. You are a customer-facing assistant, so answer the query directly addressing the user.

The generated response should answer the query directly. If you think that the query is not relevant to the documents, reply that the query is irrelevant. Provide your complete response first with all information, and then provide the citations at the end.

Format citations as:
**Citations:**
* Policy Name, Page Number
* Policy Name, Page Number
"""
                }
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            
            self.logger.info("Successfully generated LLM response")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response to your query. Please try again later."
    
    def format_context(self, search_results: List[SearchResult]) -> str:
        """
        Format search results into context for LLM.
        
        Args:
            search_results: List of search results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_part = f"""
Result {i}:
Content: {result.content}
Source: {result.metadata.policy_name}, {result.metadata.page_no}
Relevance Score: {result.rerank_score:.4f}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def extract_citations(self, search_results: List[SearchResult]) -> List[Citation]:
        """
        Extract and format citations from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of Citation objects
        """
        try:
            # Group results by policy name
            policy_pages = {}
            
            for result in search_results:
                policy_name = result.metadata.policy_name
                page_no = result.metadata.page_no
                
                if policy_name not in policy_pages:
                    policy_pages[policy_name] = []
                
                if page_no not in policy_pages[policy_name]:
                    policy_pages[policy_name].append(page_no)
            
            # Create Citation objects
            citations = []
            for policy_name, page_numbers in policy_pages.items():
                citation = Citation(
                    policy_name=policy_name,
                    page_numbers=sorted(page_numbers)
                )
                citations.append(citation)
            
            self.logger.info(f"Extracted {len(citations)} citations")
            return citations
            
        except Exception as e:
            self.logger.error(f"Error extracting citations: {e}")
            return []
    
    def parse_response_and_citations(self, full_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to separate main content and citations.
        
        Args:
            full_response: Full response from LLM
            
        Returns:
            Dictionary with 'response' and 'citations_text' keys
        """
        try:
            # Split response at citations marker
            if "**Citations:**" in full_response:
                parts = full_response.split("**Citations:**")
                response_text = parts[0].strip()
                citations_text = "**Citations:**" + parts[1].strip() if len(parts) > 1 else ""
            else:
                response_text = full_response.strip()
                citations_text = ""
            
            return {
                "response": response_text,
                "citations_text": citations_text
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing response and citations: {e}")
            return {
                "response": full_response,
                "citations_text": ""
            }
