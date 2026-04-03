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

        self.collection_name = config.collection_name
        self.directory = config.persist_directory

    def _get_or_create_collection(self, name: str = None) -> chromadb.Collection:
        """
        Gets an existing collection or creates a new one.

        args:
        - name (str): name of collection

        returns:
        - a ChromaDB Collection instance
        """
        collection_name = getattr(self.config, 'collection_name', 'documents')

        if name is None:
            col_name = collection_name
        else:
            col_name = name

        return self.client.get_or_create_collection(name=col_name) 

    def add_documents(
            self, 
            documents: Optional[List[str]] = None, 
            embeddings: Optional[List[List[float]]] = None, 
            metadatas: Optional[List[dict]] = None
        ) -> None:
        """
        Adds documents to the vector database.
        
        args:
        - documents (Optional[List[str]]): a list of text strings (documents) to add to the database
        - embeddings (Optional[List[List[float]]]): a list of text embeddings corresponding to some text strings
        - metadatas (Optional[List[Dict]]): a list of dictionaries containing document metadata
        """
        target = documents if documents else embeddings             # checks for whether the add request is valid

        if not target:
            print("Documents and embeddings cannot both be empty!")
            return
        
        import os
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]       # document identifiers
        else:
            ids = []
            for i, metadata in enumerate(metadatas):
                source_file = metadata.get("source_file", "unknown")
                chunk_index = metadata.get("chunk_index", i)
                doc_id = f"{source_file}__chunk_{chunk_index}"
                ids.append(doc_id)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(
            self, 
            query: str | List[float], 
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

        if isinstance(query, list):
            # query is an embedding
            if all(isinstance(x, float) for x in query):
                return self.collection.query(
                    query_embeddings=[query],
                    n_results=n_results,
                    include=include
                )
            elif len(query) == 1 and isinstance(query[0], list) and all(isinstance(x, float) for x in query[0]):
                return self.collection.query(
                    query_embeddings=query,
                    n_results=n_results,
                    include=include
                )
        
        else:
            # query is text only
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