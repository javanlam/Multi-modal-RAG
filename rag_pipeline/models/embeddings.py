from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    """Handles text embedding generation"""
    
    def __init__(self, config):
        self.model = SentenceTransformer(config.embedding_model)
        self.dimension = config.embedding_dimension
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        return self.model.encode(texts).tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        return self.model.encode([text])[0].tolist()