import os
from dataclasses import dataclass
from typing import Self

@dataclass
class RAGConfig:
    """
    Configuration settings for the RAG setup.
    
    Defines:
    - retrieval mode to use ("vector" for vector embedding, or "graph" for graph storage)
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
    # indicates whether to use vector embedding mode or graph storage mode
    # vector embedding mode: "vector"
    # graph storage mode: "graph"
    retrieval_mode: str = "vector"

    # preprocessing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    extract_images: bool = True
    generate_image_captions: bool = True
    image_storage_dir: str = "./image_store"
    vlm_model: str = "gpt-4o-mini"
    max_image_context_length: int = 500
    image_caption_prompt: str = """Provide a concise, descriptive caption for this image based on the surrounding text context."""

    # LLM client
    llm_provider: str = "openai-azure"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # for vector embedding mode
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
        
    persist_directory: str = "./chroma_db"
    collection_name: str = "documents"
    
    # for graph storage mode
    graph_persist_directory: str = "./graph_db"
    min_community_size: int = 3
    community_summary_length: int = 200
    entity_extraction_temperature: float = 0.1

    # retrieval
    top_k: int = 3
    similarity_threshold: float = 0.7

    @classmethod
    def from_env(cls) -> Self:
        """
        Create a config class instance from environment variables.
        Parameters may be stored in .env.
        """
        return cls(
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "vector"),
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
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            graph_persist_directory=os.getenv("GRAPH_PERSIST_DIRECTORY", "./graph_db"),
            min_community_size=int(os.getenv("MIN_COMMUNITY_SIZE", "3")),
            community_summary_length=int(os.getenv("COMMUNITY_SUMMARY_LENGTH", "200")),
            entity_extraction_temperature=float(os.getenv("ENTITY_EXTRACTION_TEMPERATURE", "0.1"))
        )