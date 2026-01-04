import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """
    Configuration settings for the RAG setup.
    
    Defines:
    - text embedding model (pretrained embedding model from SentenceTransformers)
    - text embedding vector dimension
    - chunk length for each chunk of document text
    - overlap length between two chunks
    - number of documents to retrieve on query (top k most similar items)
    - text embedding cosine similarity threshold for retrieval
    - directory to store processed documents
    - name of vector database collection
    - LLM provider, model, and parameters
    """
    
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    top_k: int = 3
    similarity_threshold: float = 0.7
    
    persist_directory: str = "./chroma_db"
    collection_name: str = "documents"
    
    llm_provider: str = "openai-azure"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls):
        """
        Create a config class instance from environment variables.
        Parameters may be stored in .env.
        """
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            persist_directory=os.getenv("PERSIST_DIRECTORY", "./chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "documents"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai-azure"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.1"))
        )