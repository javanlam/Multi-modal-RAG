from typing import List, Dict, Any, Optional
from config.settings import RAGConfig
from models.multimodal_embeddings import MultimodalEmbeddingModel
from .graph_store import GraphStorageManager
from .generator import ResponseGenerator
from .image_store import ImageStore
from .multi_vector_store import MultiVectorStoreManager


class GraphRetriever:
    """
    Graph-based document retriever using knowledge graphs.
    Handles both entity-specific and global questions.
    """
    
    def __init__(
            self, 
            graph_store: GraphStorageManager, 
            config: RAGConfig, 
            generator: ResponseGenerator,
            image_store: ImageStore,
            multi_vector_store: MultiVectorStoreManager,
            multimodal_embedding: MultimodalEmbeddingModel
        ):
        """
        Initializes an instance of the graph retriever with configurations provided.

        args:
        - graph_store (GraphStorageManager): an instance of the graph storage manager
        - config (RAGConfig): an instance of the data class for configuration settings
        - generator (ResponseGenerator): generator to obtain enhanced query
        - image_store (ImageStore): storage of extracted imgaes
        - multi_vector_store (MultiVectorStoreManager): multi-vector database of image feature and caption embeddings
        - multimodal_embedding (MultimodalEmbeddingModel): multimodal embedding model
        """
        self.graph_store = graph_store
        self.config = config
        self.generator = generator
        self.image_store = image_store
        self.multi_vector_store = multi_vector_store
        self.multimodal_embedding = multimodal_embedding
    
    def _classify_question_type(self, question: str) -> str:
        """
        Classifies whether question is entity-specific or global.
        
        args:
        - question (str): question from the user

        returns:
        - type of question ("entity" for searching by entity, "global" for global/sensemaking questions)
        """
        global_indicators = [
            "overall", "generally", "in general", "main themes",
            "key takeaways", "broadly", "entire dataset", "whole collection",
            "summarize", "overview", "what are the main", "what can you tell me about"
        ]

        question_lower = question.lower()

        for indicator in global_indicators:
            if indicator in question_lower:
                return "global"
            
        return "entity"
    
    def _enhance_query(self, query: str) -> str:
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
    
    def retrieve(self, question: str, use_enhancement: bool = True) -> Dict[str, Any]:
        """
        Retrieves relevant information using graph-based approach.
        
        args:
        - question (str): question from the user
        - use_enhancement (bool): whether to enhance user prompt

        returns:
        - a dictionary containing retrieved documents and retrieval metadata
        """
        question_type = self._classify_question_type(question=question)

        if question_type == "global":
            if use_enhancement:
                question_enhanced = self._enhance_query(query=question)
            else:
                question_enhanced = question

            results = self.graph_store.search_global_question(question=question_enhanced)

            context_docs = []
            for community in results["communities"]:
                context_docs.append(
                    f"Community {community['community_id']}: {community['community_summary']}\n"
                    f"Key entities: {', '.join(community['entities'])}"
                )

            retrieval_results = {
                "documents": context_docs,
                "original_query": question,
                "enhanced_query": question_enhanced,
                "question_type": "global",
                "metadata": results
            }

            return retrieval_results

        else:
            results = self.graph_store.search_by_entity(query=question)

            context_docs = []

            for entity in results["matching_entities"][:3]:
                context_docs.append(
                    f"Entity: {entity['name']}\n"
                    f"Description: {entity['description']}\n"
                    f"Connections: {entity['degree']} relationships"
                )

            for community in results["relevant_communities"]:
                context_docs.append(
                    f"Related Community: {community['summary']}\n"
                    f"Example entities: {', '.join(community['entities'])}"
                )

            retrieval_results = {
                "documents": context_docs,
                "original_query": question,
                "enhanced_query": question,
                "question_type": "entity",
                "metadata": results
            }

            return retrieval_results
        
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