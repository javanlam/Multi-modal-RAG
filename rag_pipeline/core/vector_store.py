import chromadb
from typing import List, Optional, Dict, Literal
from config.settings import RAGConfig


class VectorStoreManager:
    """
    Manages vector database operations using ChromaDB.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the vector database class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        self.collection = self._get_or_create_collection()
    

    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Gets an existing collection or creates a new one.

        returns:
        - a ChromaDB Collection instance
        """
        collection_name = getattr(self.config, 'collection_name', 'documents')
        return self.client.get_or_create_collection(name=collection_name)
    

    def add_documents(
            self, 
            documents: List[str], 
            embeddings: Optional[List[List[float]]] = None, 
            metadatas: Optional[List[dict]] = None
        ) -> None:
        """
        Adds documents to the vector database.
        
        args:
        - documents (List[str]): a list of text strings (documents) to add to the database
        - embeddings (List[List[float]]): a list of text embeddings corresponding to some text strings
        - metadatas (Optional[List[Dict]]): a list of dictionaries containing document metadata
        """
        target = documents if documents else embeddings             # checks for whether the add request is valid

        if not target:
            print("Documents and embeddings cannot both be empty!")
            return
        
        import os
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        ids = [f"doc_{i}" for i in range(len(documents))]           # document identifiers
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    

    def search(
            self, 
            query: str, 
            n_results: int = None,
            include: set[Literal["documents", "embeddings", "metadatas", "distances"]] = None
        ) -> Dict:
        """
        Searches for similar documents based on a given query.
        
        args:
        - query (str | List[float]): query to search for (can be a raw string or text embeddings)
        - n_results (int): number of results to return
        - include (set): a set of items to include in query output

        returns:
        - a dictionary containing the retrieved items
        """
        if n_results is None:
            n_results = self.config.top_k

        if include is None:
            include = ["documents", "metadatas", "distances"]

        include = list(include)
        
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=include
        )
    

    def get_collection_info(self) -> Dict:
        """
        Gets information about the collection.
        
        returns:
        - a dictionary containing information about the collection
        """
        collection_info = {
            "name": self.collection_name,
            "count_items": self.collection.count(),
            "directory": self.directory
        }

        return collection_info