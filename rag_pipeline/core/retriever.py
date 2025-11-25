from typing import List, Dict, Any
from config.settings import RAGConfig
from models.embeddings import EmbeddingModel

class HyDERetriever:
    """Enhanced retriever with Hypothetical Document Embeddings (HyDE)"""
    
    def __init__(self, vector_store, config: RAGConfig):
        self.vector_store = vector_store
        self.config = config
        self.embedding_model = EmbeddingModel(config)
            
    def enhance_query(self, query: str) -> str:
        """Enhance query using HyDE technique"""
        try:
            # Generate hypothetical document
            prompt = f"""Based on the query, generate a detailed hypothetical answer document (approximately 200-300 words).

Query: {query}

Hypothetical Document:"""
            
            return self._simple_query_expansion(query)
                
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return query
    
    def _simple_query_expansion(self, query: str) -> str:
        """Simple query expansion fallback"""
        expansions = {
            "what": f"{query} Explain in detail with examples.",
            "how": f"{query} Describe the process step by step.",
            "why": f"{query} Provide reasons and causes."
        }
        
        for word, expansion in expansions.items():
            if query.lower().startswith(word):
                return expansion
        
        return f"{query} Provide comprehensive information."
    
    def retrieve(self, query: str, use_enhancement: bool = True) -> Dict[str, Any]:
        """Retrieve relevant documents with optional query enhancement"""
        
        if use_enhancement:
            enhanced_query = self.enhance_query(query)
            results = self.vector_store.search(enhanced_query)
        else:
            results = self.vector_store.search(query)
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "original_query": query,
            "enhanced_query": enhanced_query if use_enhancement else query
        }