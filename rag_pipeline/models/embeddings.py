from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List
from config.settings import RAGConfig


class EmbeddingModel:
    """
    Handles text embedding generation with a specified pretrained embedding model.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the embedding model class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.model = SentenceTransformer(config.embedding_model)
        self.dimension = config.embedding_dimension

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for texts.
        
        args:
        - texts (List[str]): the list of texts to generate embeddings for

        returns:
        - a list of vector embeddings corresponding to each text item
        """
        return self.model.encode(texts).tolist()

    def encode_single(self, text: str) -> List[float]:
        """
        Generates embedding for single text string.
        
        args:
        - text (str): the text to generate embeddings for

        returns:
        - vector embeddings corresponding to the provided text
        """
        return self.model.encode([text])[0].tolist()
    
    def embedding_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Computes the cosine similarity between two text embeddings.

        args:
        - embedding1 (List[float]): first text embedding
        - embedding2 (List[float]): second text embedding

        returns:
        - the cosine similarity of the two embeddings as a floating point number 
        """
        similarity = util.cos_sim(embedding1, embedding2)       # cos_sim() returns a tensor

        return similarity.item()                                # Tensor.item() to the floating point number inside