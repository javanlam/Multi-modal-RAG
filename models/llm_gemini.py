import os
from pathlib import Path
import dotenv
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from config.settings import RAGConfig
from utils.image_extraction import data_url_to_bytes


class LLM_Gemini:
    """
    Provides functionality for generating responses from Google Gemini LLM.
    """

    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config

        self.provider = self.config.llm_provider
        self.model = self.config.llm_model
        self.temperature = self.config.temperature

        self._setup_llm()

    def _setup_llm(self) -> None:
        """
        Sets up the LLM based on configuration.
        """
        dotenv_path = Path(__file__).parent.parent / ".env"
        dotenv.load_dotenv(dotenv_path=dotenv_path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)

        self.llm_client = genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": getattr(self.config, "max_tokens", 1000),
            }
        )

    def _get_system_prompt(self) -> str:
        """
        Returns a generic system prompt when no system prompt is provided.

        returns:
        - the system prompt
        """
        system_prompt = """You are a helpful assistant that follows closely the following instructions provided by the user."""

        return system_prompt

    def generate_response(self, user_prompt: str, system_prompt: Optional[str] = None, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generates a response from the LLM.

        args:
        - user_prompt (str): user prompt containing instructions and the query
        - system_prompt (Optional[str]): optional system prompt containing global context and instructions
        - images (Optional[List[str]]): optional list of images in data URL form to pass to the LLM
        
        returns:
        - a dictionary containing the generated response and additional information
        """
        if system_prompt is None:
            system_prompt = self._get_system_prompt()

        try:
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": getattr(self.config, "max_tokens", 1000),
                }
            )
            parts = [user_prompt]

            if images:
                for img_data_url in images:
                    try:
                        image_bytes, mime_type = data_url_to_bytes(img_data_url)
                        parts.append({
                            "mime_type": mime_type,
                            "data": image_bytes
                        })
                    except ValueError:
                        # assume local file path or URL
                        parts.append(img_data_url)

            response = model.generate_content(parts)

            if not response.candidates:
                return {"answer": "No response generated (blocked).", "error": True}
            
            return {
                "answer": response.text,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                },
                "model": self.model
            }

        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "error": True
            }