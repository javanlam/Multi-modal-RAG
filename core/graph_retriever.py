from typing import List, Dict, Any
from config.settings import RAGConfig
from core.graph_store import GraphStorageManager
from core.generator import ResponseGenerator


class GraphRetriever:
    """
    Graph-based document retriever using knowledge graphs.
    Handles both entity-specific and global questions.
    """
    
    def __init__(self, graph_store: GraphStorageManager, config: RAGConfig):
        """
        Initializes an instance of the graph retriever.

        args:
        - graph_store (GraphStorageManager): an instance of the graph storage manager
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.graph_store = graph_store
        self.config = config
        self.generator = ResponseGenerator(config=config)
    
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
            
            enhanced_prompt_output = self.generator.generate_openai_response(prompt=prompt, query="")

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