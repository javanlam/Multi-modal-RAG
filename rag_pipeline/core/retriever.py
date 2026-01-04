import chromadb
from typing import List, Dict, Any
from config.settings import RAGConfig
from models.embeddings import EmbeddingModel
from generator import ResponseGenerator


class HyDERetriever:
    """
    Enhanced document retriever with Hypothetical Document Embeddings (HyDE).
    Helps generalize to other unseen domains.
    Details can be found at https://aclanthology.org/2023.acl-long.99.pdf
    """
    
    def __init__(self, vector_store: chromadb.Collection, config: RAGConfig):
        """
        Initializes an instance of the retriever with configurations provided.

        args:
        - vector_store (chromadb.Collection): a ChromaDB collection that acts as the external knowledge base
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.vector_store = vector_store
        self.config = config
        self.embedding_model = EmbeddingModel(config)


    def enhance_query(self, query: str) -> str:
        """
        Enhances user query using the HyDE technique.
        
        args:
        - query (str): user query to enhance

        returns:
        - a string containing the enhanced query
        """
        try:
            # generates hypothetical document
            prompt = f"""Based on the query, generate a detailed hypothetical answer document (approximately 200-300 words).

Query: {query}

Hypothetical Document:
"""
            
            generator = ResponseGenerator(config=self.config)
            enhanced_prompt_output = generator.generate_openai_response(prompt=prompt, query="")

            enhanced_prompt = enhanced_prompt_output.get("answer", query)

            return enhanced_prompt
                
        except Exception as e:
            # catch error in using an LLM to generate enhanced prompt; use rule-based fallback option
            return self._simple_query_expansion(query)
    

    def _simple_query_expansion(self, query: str) -> str:
        """
        Expands user query, acts as a fallback option when the HyDE technique fails.
        
        args:
        - query (str): user query to expand

        returns:
        - a string containing the expanded query
        """
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
        """
        Retrieves relevant documents with optional query enhancement.
        
        args:
        - query (str): user query to retrieve documents for
        - use_enhancement (bool): whether to enhance user prompt

        returns:
        - a dictionary containing the query, enhanced query, and retrieved documents
        """
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