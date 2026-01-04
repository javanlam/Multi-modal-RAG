import os
from pathlib import Path
import dotenv
import openai
from typing import List, Dict, Any
from config.settings import RAGConfig


class ResponseGenerator:
    """
    Generates responses by passing query and retrieved context to an LLM.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the generator with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config
        self._setup_llm()
    

    def _setup_llm(self) -> None:
        """
        Sets up the LLM based on configuration.
        """
        dotenv_path = Path(__file__).parent.parent / ".env"
        dotenv.load_dotenv(dotenv_path=dotenv_path)

        if self.config.llm_provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.llm_client = None

        elif self.config.llm_provider == "openai-azure":
            self.llm_client = openai.AzureOpenAI(
                api_key = os.getenv("OPENAI_API_KEY"),
                api_version = "2024-06-01",
                azure_endpoint = "https://hkust.azure-api.net",
                azure_deployment = self.config.llm_model
            )
    
    
    def generate_response(self, query: str, context_documents: List[str]) -> Dict[str, Any]:
        """
        Generates a response using context and query.
        
        args;
        - query (str): user query to generate a response to
        - context_documents (List[str]): a list of documents retrieved from external knowledge base, to act as context

        returns:
        - a dictionary containing the generated response and additional information
        """
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_documents)])
        
        prompt = self._build_prompt(query, context)
        
        if self.config.llm_provider == "openai" or self.config.llm_provider == "openai-azure":
            return self.generate_openai_response(prompt, query)
        
        else:
            return self._generate_fallback_response(query, context)
    

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Builds a prompt for the LLM.
        
        args:
        - query (str): user query to generate a response to
        - context (str): retrieved information from the external database to be used as context

        returns:
        - a string containing the prompt to the LLM.
        """
        return f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, state that clearly.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based solely on the context provided. If the context is insufficient, explain what information is missing.

Answer:"""
    

    def generate_openai_response(self, prompt: str, query: str) -> Dict[str, Any]:
        """
        Generates a response using OpenAI API.
        
        args:
        - prompt (str): prompt containing the user's query and retrieved context
        - query (str): user query to generate a response to

        returns:
        - a dictionary containing the generated response and additional information
        """
        try:

            if self.llm_client is None:
                response = openai.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=1000
                )

            else:
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=1000
                )
            
            return {
                "answer": response.choices[0].message.content,
                "usage": response.usage,
                "model": self.config.llm_model
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "error": True
            }
    

    def _generate_fallback_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generates a response when an error is caught when calling the LLM.
        
        args:
        - query (str): user query to generate a response to
        - context (str): retrieved information from the external database to be used as context

        returns:
        - a dictionary containing the fallback response and additional information
        """
        return {
            "answer": f"Based on the context, I can provide information about: {query}. Context length: {len(context)} characters.",
            "fallback": True
        }