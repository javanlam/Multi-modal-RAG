import os
from pathlib import Path
import dotenv
import openai
from typing import List, Dict, Any, Optional
from config.settings import RAGConfig
from models.llm_gemini import LLM_Gemini
from models.llm_openai import LLM_OpenAI
from models.llm_openai_azure import LLM_OpenAI_Azure
from models.llm_qwen import LLM_Qwen


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
            self.llm = LLM_OpenAI(self.config)

        elif self.config.llm_provider == "openai-azure":
            self.llm = LLM_OpenAI_Azure(self.config)

        elif self.config.llm_provider == "google":
            self.llm = LLM_Gemini(self.config)

        elif self.config.llm_provider == "qwen":
            self.llm = LLM_Qwen(self.config)

        else:
            self.llm = LLM_OpenAI_Azure(self.config)
    
    def generate_response(
            self, 
            query: str, 
            context_documents: str, 
            query_img: Optional[List[str]] = None,
            context_images: Optional[List[str]] = None
        ) -> Dict[str, Any]:
        """
        Generates a response using context and query.
        
        args:
        - query (str): user query to generate a response to
        - context_documents (List[str]): a list of documents retrieved from external knowledge base, to act as context
        - query_img (Optional[List[str]]): a list of encoded image URLs in the user's query
        - context_images (Optional[List[str]]): a list of encoded image URLs for images retrieved from external knowledge base, to act as context

        returns:
        - a dictionary containing the generated response and additional information
        """
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_documents)])
        
        prompt = self._build_prompt(query, context, query_img, context_images)
        system_prompt = "You are a helpful assistant that follows closely the following instructions provided by the user."

        images = None

        if query_img or context_images:
            images = []

            if query_img:
                images.extend(query_img)
            
            if context_images:
                images.extend(context_images)

        try:
            return self.llm.generate_response(user_prompt=prompt, system_prompt=system_prompt, images=images)
        except:
            return self._generate_fallback_response(query, context)

    def _build_prompt(
            self, 
            query: str, 
            context: str, 
            query_img: Optional[List[str]] = None,
            context_images: Optional[List[str]] = None
        ) -> str:
        """
        Builds a prompt for the LLM.
        
        args:
        - query (str): user query to generate a response to
        - context (str): retrieved information from the external database to be used as context
        - query_img (Optional[List[str]]): a list of encoded image URLs in the user's query
        - context_images (Optional[List[str]]): a list of encoded image URLs for images retrieved from external knowledge base, to act as context

        returns:
        - a string containing the prompt to the LLM.
        """
        prompt = f"""You are a helpful assistant that answers questions based on retrieved context in possibly both text and visual form.

{"" if not query_img else f"The user has provided an image in their query, and it is the first {len(query_img)} images among all images presented to you."}
{"" if not context_images else f"The last {len(context_images)} presented to you are relevant visual context to assist you in answering the question."}   

Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer.
Always trust the context over your own knowledge.
If you find the context insufficient for answering the question, gently reject answering the question.

You act as a chatbot and are having a conversation with the user.
Provide a natural response as in a conversation; do NOT mention anything about the context in your answer.

Answer:"""
        
        return prompt

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

            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                {"role": "user", "content": prompt}
            ]

            if self.llm_client is None:
                response = openai.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=1000
                )

            else:
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
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