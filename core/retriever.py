import chromadb
from typing import List, Dict, Any, Optional, Tuple
from config.settings import RAGConfig
from models.embeddings import EmbeddingModel
from models.multimodal_embeddings import MultimodalEmbeddingModel
from .generator import ResponseGenerator
from .image_store import ImageStore
from .multi_vector_store import MultiVectorStoreManager
from .vector_store import VectorStoreManager


class HyDERetriever:
    """
    Enhanced document retriever with Hypothetical Document Embeddings (HyDE).
    """
    
    def __init__(
            self, 
            vector_store: VectorStoreManager, 
            config: RAGConfig, 
            generator: ResponseGenerator, 
            image_store: ImageStore,
            multi_vector_store: MultiVectorStoreManager
        ):
        """
        Initializes an instance of the retriever with configurations provided.

        args:
        - vector_store (VectorStoreManager): a VectorStoreManager object that acts as the external knowledge base
        - config (RAGConfig): an instance of the data class for configuration settings
        - generator (ResponseGenerator): generator to obtain enhanced query
        - image_store (ImageStore): image store object to retrieve from
        - multi_vector_store (MultiVectorStoreManager): multi-vector database containing embedded images and captions
        """
        self.vector_store = vector_store
        self.config = config
        self.embedding_model = EmbeddingModel(self.config)
        self.generator = generator
        self.multimodal_embedding = MultimodalEmbeddingModel(self.config)
        self.image_store = image_store
        self.multi_vector_store = multi_vector_store

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
            
            enhanced_prompt_output = self.generator.llm.generate_response(user_prompt=prompt)

            enhanced_prompt = enhanced_prompt_output.get("answer", query)

            if "Error generating response" in enhanced_prompt:
                enhanced_prompt = query

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
    
    def retrieve_multimodal(self, query: str, query_images: Optional[List[str]]) -> List[str]:
        """
        Retrieves relevant images.

        args:
        - query (str): user query to retrieve images for
        - query_images (Optional[List[str]]): list of data URLs of images included in the user's query

        returns:
        - a list of image data URLs of retrieved images
        """
        retrieved_image_ids = []
        retrieved_image_data_urls = []

        try:
            # search by text
            query_emb_text = self.multimodal_embedding.encode_query(text=query)
            text_to_caption_results, text_to_image_results = self.multi_vector_store.search_by_caption(query_emb_text, n_results=self.config.top_k)

            # search by images (if any)
            query_emb_imgs = []
            by_image_results = []
            
            if query_images:
                query_emb_img = self.multimodal_embedding.encode_query(image_data_url=query_images[0])
                query_emb_imgs.append(query_emb_img)

                for query_img in query_emb_imgs:
                    by_image_result = self.multi_vector_store.search_by_image(query_img, n_results=self.config.top_k)
                    by_image_results.append(by_image_result)

            if text_to_caption_results:
                retrieved_image_ids.extend(text_to_caption_results['ids'][0])

            if text_to_image_results:
                retrieved_image_ids.extend(text_to_image_results['ids'][0])

            if by_image_results:
                retrieved_image_ids.extend(by_image_result['ids'][0] for by_image_result in by_image_results)

            # remove duplicates
            retrieved_image_ids = list(set(retrieved_image_ids))
            
            # retrieve image URLs from image store
            retrieved_image_data_urls = self.image_store.get_image_data_urls(retrieved_image_ids)

            print(f"Retrieved {len(retrieved_image_data_urls)} images via multi-modal search")

        except Exception as e:
            print(f"Multi-modal retrieval error: {e}")

        return retrieved_image_data_urls