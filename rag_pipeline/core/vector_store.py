import chromadb
from typing import List, Optional
from config.settings import RAGConfig

class VectorStoreManager:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one"""
        collection_name = getattr(self.config, 'collection_name', 'documents')
        return self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to the vector store"""
        if not documents:
            return
        
        import os
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        ids = [f"doc_{i}" for i in range(len(documents))]   # document identifiers
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = None) -> dict:
        """Search for similar documents"""
        if n_results is None:
            n_results = self.config.top_k
        
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def get_collection_info(self) -> dict:
        """Get information about the collection"""
        return self.collection.count()