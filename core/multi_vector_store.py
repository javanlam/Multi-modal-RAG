import chromadb
from typing import List, Dict, Optional, Literal, Union, Tuple
from config.settings import RAGConfig


class MultiVectorStoreManager:
    """
    Manages two vector database operations using ChromaDB.
    """

    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the multi-vector database class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        self.image_collection = self._get_or_create_collection(name=f"{config.collection_name}_{config.image_embeddings_collection}")
        self.caption_collection = self._get_or_create_collection(name=f"{config.collection_name}_{config.caption_embeddings_collection}")

        self.collection_name = f"{config.collection_name}_images"
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

    def add_image(
            self,
            image_id: str,
            image_embedding: List[float],
            caption_embedding: List[float],
            metadata: Optional[Dict] = None,
        ) -> None:
        """
        Adds one image-caption pair to the collections.

        args:
        - image_id (str): identifier for the image
        - image_embedding (List[float]): embedding vector obtained from DINOv2
        - caption_embedding (List[float]): embedding vector of the caption
        - metadata (Optional[Dict]): dictionary containing metadata of the image
        """
        if metadata is None:
            metadata = {"source": "unknown"}

        self.image_collection.add(
            embeddings=[image_embedding],
            metadatas=[metadata],
            ids=[image_id],
        )

        self.caption_collection.add(
            embeddings=[caption_embedding],
            metadatas=[metadata],
            ids=[image_id],
        )

    def add_image_batch(
            self,
            image_ids: List[str],
            image_embeddings: List[List[float]],
            caption_embeddings: List[List[float]],
            metadatas: Optional[List[Dict]] = None,
        ) -> None:
        """
        Adds a batch of multiple image-caption pairs to the collections.

        args:
        - image_ids (List[str]): a list containing identifiers for the images
        - image_embeddings (List[List[float]]): a list of embedding vectors obtained from DINOv2
        - caption_embeddings (List[List[float]]): a list of embedding vectors of image captions
        - metadatas (Optional[List[Dict]]): a list of dictionaries containing image metadata dicts
        """
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in image_ids]

        self.image_collection.add(
            embeddings=image_embeddings,
            metadatas=metadatas,
            ids=image_ids,
        )

        self.caption_collection.add(
            embeddings=caption_embeddings,
            metadatas=metadatas,
            ids=image_ids,
        )

    def search_by_image(
            self,
            query_embedding: List[float],
            n_results: int = None,
            include: Optional[set[Literal["metadatas", "distances"]]] = None,
        ) -> Dict:
        """
        Searches for images by embedding similarity.

        args:
        - query_embedding (List[float]): embedding vector of a query image in DINOv2 space
        - n_results (int): number of results to return
        - include (set): a set of items to include in query output

        returns:
        - a dictionary containing information about the retrieved items
        """
        if include is None:
            include = ["metadatas", "distances"]

        include = list(include)

        if n_results is None:
            n_results = self.config.top_k

        return self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include,
        )

    def search_by_caption(
            self,
            query_embedding: List[float],
            n_results: int = None,
            include: Optional[set[Literal["metadatas", "distances"]]] = None,
        ) -> Tuple[Dict, Dict]:
        """
        Searches for images by embedding similarity between query text and image embeddings.

        args:
        - query_embedding (List[float]): embedding vector of a text query
        - n_results (int): number of results to return
        - include (set): a set of items to include in query output

        returns:
        - a tuple of two dictionaries containing information about the retrieved items
        """
        if include is None:
            include = ["metadatas", "distances"]

        include = list(include)

        if n_results is None:
            n_results = self.config.top_k

        by_captions = self.caption_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include,
        )

        by_cross_modal = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include,
        )

        return by_captions, by_cross_modal

    def get_collection_info(self) -> Dict:
        """
        Gets information about the collection.
        
        returns:
        - a dictionary containing information about the collection
        """
        return {
            "image_collection": self.image_collection.name,
            "caption_collection": self.caption_collection.name,
            "image_count": self.image_collection.count(),
            "caption_count": self.caption_collection.count(),
        }