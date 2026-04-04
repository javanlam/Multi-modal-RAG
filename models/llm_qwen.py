import os
from pathlib import Path
import dotenv
from dashscope import MultiModalConversation
from typing import List, Dict, Any, Optional
from config.settings import RAGConfig


class LLM_Qwen:
    """
    Provides functionality for generating responses from Alibaba Qwen LLM.
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

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables")

        os.environ["DASHSCOPE_API_KEY"] = api_key

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
            content = [{"text": user_prompt}]

            if images:
                for img_url in images:
                    content.append({"image": img_url})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]

            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=getattr(self.config, "max_tokens", 1000)
            )

            if response.status_code == 200:
                answer = response.output.choices[0].message.content[0]["text"]
                usage = response.usage
                return {
                    "answer": answer,
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "model": self.model
                }
            else:
                return {
                    "answer": f"Error: {response.code} - {response.message}",
                    "error": True
                }

        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "error": True
            }