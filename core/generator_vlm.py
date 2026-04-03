import os
import dotenv
import openai
from typing import List, Dict, Any, Optional
from config.settings import RAGConfig
from .generator import ResponseGenerator


class ResponseGeneratorVLM(ResponseGenerator):
    """
    Generates responses by passing query and retrieved context to an LLM.
    This version of the class adds support for visual context.
    """

    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the generator with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        super().__init__(config)

    def generate_response(
            self,
            query: str,
            context_documents: List[str],
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

        if self.config.llm_provider == "openai" or self.config.llm_provider == "openai-azure":
            return self.generate_openai_response(prompt, query, query_img, context_images)
        
        else:
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

Please provide a comprehensive answer based on the context provided.
If you find the context insufficient for answering the question, gently reject answering the question.

You act as a chatbot and are having a conversation with the user.
Provide a natural response as in a conversation.

Answer:"""
        
        return prompt

    def generate_openai_response(
            self, 
            prompt: str, 
            query: str,
            query_img: Optional[List[str]] = None,
            context_images: Optional[List[str]] = None
        ) -> Dict[str, Any]:
        """
        Generates a response using OpenAI API.
        
        args:
        - prompt (str): prompt containing the user's query and retrieved context
        - query (str): user query to generate a response to
        - query_img (Optional[List[str]]): a list of encoded image URLs in the user's query
        - context_images (Optional[List[str]]): a list of encoded image URLs for images retrieved from external knowledge base, to act as context

        returns:
        - a dictionary containing the generated response and additional information
        """
        try:

            image_data = []

            if query_img:       # user query images
                image_data.extend([{
                    "type": "image_url",
                    "image_url": {"url": img}
                } for img in query_img])

            if context_images:  # retrieved context images
                image_data.extend([{
                    "type": "image_url",
                    "image_url": {"url": img}
                } for img in context_images])

            user_messages = [{
                "type": "text",
                "text": prompt
            }]

            if len(image_data) > 0:
                user_messages.extend(image_data)

            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                {"role": "user", "content": user_messages if len(user_messages) > 1 else prompt}
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