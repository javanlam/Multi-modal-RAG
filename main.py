import os
import base64
from typing import Dict, List, Tuple, Optional
from config.settings import RAGConfig
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.multi_vector_store import MultiVectorStoreManager
from core.graph_store import GraphStorageManager
from core.retriever import HyDERetriever
from core.graph_retriever import GraphRetriever
from core.generator import ResponseGenerator
from core.generator_vlm import ResponseGeneratorVLM
from core.image_store import ImageStore
from models.embeddings import EmbeddingModel
from models.multimodal_embeddings import MultimodalEmbeddingModel

class RAGSystem:
    """
    Main RAG system class that sets up all components.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initializes an instance of the RAG System class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config or RAGConfig.from_env()

        self.vlm_generator = ResponseGeneratorVLM(self.config) if config.generate_image_captions else None
        
        if self.config.extract_images:
            self.image_store = ImageStore(self.config.image_store_dir)
            print(f"Image store initialized at: {self.config.image_store_dir}")
        else:
            self.image_store = None
        
        self.document_processor = DocumentProcessor(self.config, self.vlm_generator, self.image_store)

        self.multimodal_embedding = None
        if self.config.use_multimodal:
            try:
                self.multimodal_embedding = MultimodalEmbeddingModel(self.config)
                print("Multi‑modal embedding model (DINOv2+Talk2DINO) loaded.")

                self.multi_vector_store = MultiVectorStoreManager(self.config)
            except Exception as e:
                print(f"Failed to load multi‑modal embedding model: {e}")
                self.config.use_multimodal = False      # prevent errors

        if self.config.retrieval_mode == "vector":
            self.vector_store = VectorStoreManager(self.config)
            self.retriever = HyDERetriever(self.vector_store, self.config)

        elif self.config.retrieval_mode == "graph":
            self.graph_store = GraphStorageManager(self.config)
            self.retriever = GraphRetriever(self.graph_store, self.config)
            if not self.graph_store.load():
                print("No existing graph found. Will build new graph when documents are ingested.")

        self.generator = ResponseGenerator(self.config)
        self.embedding = EmbeddingModel(self.config)
    
    def ingest_documents(self, source_path: str):
        """
        Ingests documents from a file or directory.
        
        args:
        - source_path (str): path of file or directory to process
        """
        if os.path.isfile(source_path):         # processes one file
            text, images_metadata = self.document_processor.load_document(
                source_path, 
                extract_images=self.config.extract_images
            )

            text_with_captions = self.document_processor.insert_image_captions(text, images_metadata)
            
            # store extracted images 
            stored_image_ids = []
            if self.image_store and images_metadata:
                for img_meta in images_metadata:
                    stored_id = self.image_store.store_image(
                        img_meta.get("image_data_url", ""), 
                        img_meta
                    )

                    if stored_id:
                        stored_image_ids.append(stored_id)
                        # update metadata with image_id
                        img_meta["image_id"] = stored_id

                print(f"Stored {len(images_metadata)} images from {os.path.basename(source_path)}")

            chunks, metadatas = self.document_processor.chunk_text(
                text_with_captions, 
                source_file=os.path.basename(source_path), 
                images_metadata=images_metadata
            )

        else:                                   # processes a directory
            chunks, metadatas, images_metadata = self.document_processor.process_directory(source_path, self.config.extract_images)

        if self.config.retrieval_mode == "vector":
            embeddings = self.embedding.encode(chunks)
            self.vector_store.add_documents(chunks, embeddings, metadatas)
            print(f"Ingested {len(chunks)} document chunks")

            chunks_with_images = sum(1 for meta in metadatas if meta.get("has_images", False))
            if chunks_with_images > 0:
                print(f"Found {chunks_with_images} chunks containing images")

        elif self.config.retrieval_mode == "graph":
            self.graph_store.add_documents(chunks, metadatas)
            print(f"Built knowledge graph from {len(chunks)} document chunks")

            chunks_with_images = sum(1 for meta in metadatas if meta.get("has_images", False))
            if chunks_with_images > 0:
                print(f"Found {chunks_with_images} chunks containing images")

        if self.config.use_multimodal and self.multimodal_embedding and self.image_store and images_metadata:
            print("Computing DINO embeddings for extracted images...")

            image_ids = []
            image_embeddings = []
            image_metadatas = []
            caption_embeddings = []

            for img_meta in images_metadata:
                image_id = img_meta.get("image_id")

                if not image_id:
                    continue

                # get the stored image data URL from ImageStore (or from metadata)
                data_url = self.image_store.get_image_data_url(image_id)

                if not data_url:
                    continue
                
                try:
                    emb = self.multimodal_embedding.encode_image(data_url)
                    image_ids.append(image_id)
                    image_embeddings.append(emb)

                    image_caption = img_meta.get("caption", "")

                    image_metadatas.append({
                        "image_id": image_id,
                        "caption": image_caption,
                        "source_file": img_meta.get("source_file", "unknown"),
                        "page_num": img_meta.get("page_num", 0),
                        "has_caption": img_meta.get("has_caption", False)
                    })

                    caption_emb = self.embedding.encode_single(image_caption)
                    caption_embeddings.append(caption_emb)
                except Exception as e:
                    print(f"Error computing embedding for image {image_id}: {e}")
                    continue

            if image_ids:
                self.multi_vector_store.add_image_batch(image_ids, image_embeddings, caption_embeddings, image_metadatas)
                print(f"Added {len(image_ids)} image embeddings to vector store.")

    def query(
            self, 
            question: str,
            query_images: Optional[List[str]] = None, 
            use_enhancement: bool = True, 
            use_vlm: bool = False
        ) -> Dict:
        """
        Processes a query and returns the generated response.

        args:
        - question (str): user query to retrieve documents for
        - query_images (Optional[List[str]]): list of data URLs of images included in the user's query
        - use_enhancement (bool): whether to enhance user prompt
        - use_vlm (bool): whether to use VLM for generation (for when images are involved)

        returns:
        - a dictionary containing the question, the generated answer, source documents, and additional information
        """
        # text retrieval
        retrieval_result = self.retriever.retrieve(question, use_enhancement)

        # multimodal retrieval
        retrieved_image_ids = []
        retrieved_image_data_urls = []
        retrieved_has_images = False

        if self.config.use_multimodal and self.multimodal_embedding and self.multi_vector_store:
            try:
                # search by text
                query_emb_text = self.multimodal_embedding.encode_query(text=question)
                text_to_caption_results, text_to_image_results = self.multi_vector_store.search_by_caption(query_emb_text, n_results=self.config.top_k)

                # search by images (if any)
                query_emb_imgs = []
                by_image_results = []
                
                if query_images:
                    query_emb_img = self.multimodal_embedding.encode_query(image_data_url=query_images[0])
                    query_emb_imgs.append(query_emb_img)

                    for query_img in query_emb_imgs:
                        by_image_result = self.multi_vector_store.search_by_image(query_img, n_results=self.config.top_k)
                        by_image_results.append(by_image_result)

                if text_to_caption_results and text_to_image_results and by_image_results:
                    retrieved_image_ids.extend(text_to_caption_results['ids'][0])
                    retrieved_image_ids.extend(text_to_image_results['ids'][0])
                    retrieved_image_ids.extend(by_image_result['ids'][0] for by_image_result in by_image_results)

                    # remove duplicates
                    retrieved_image_ids = list(set(retrieved_image_ids))
                    
                    # retrieve image URLs from image store
                    retrieved_image_data_urls = self.get_image_data_urls(retrieved_image_ids)
                    retrieved_has_images = len(retrieved_image_data_urls) > 0

                    print(f"Retrieved {len(retrieved_image_data_urls)} images via multi‑modal search")

            except Exception as e:
                print(f"Multi‑modal retrieval error: {e}")

        for doc in retrieval_result["documents"]:
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
            elif isinstance(doc, dict) and 'metadata' in doc:
                metadata = doc['metadata']
            else:
                metadata = doc if isinstance(doc, dict) else {}
            
            if metadata.get("has_images", False):
                retrieved_has_images = True
                image_ids = metadata.get("image_ids", [])

                if image_ids and self.image_store:
                    # get data URLs for images from store
                    chunk_image_urls = self.get_image_data_urls(image_ids)
                    retrieved_image_data_urls.extend(chunk_image_urls)

        context_images = retrieved_image_data_urls
        if context_images:
            print(f"Using {len(context_images)} images as context")

        if use_vlm and self.vlm_generator and (retrieved_has_images or context_images or query_images):
            generation_result = self.vlm_generator.generate_response(
                question, 
                context_documents=retrieval_result["documents"],
                query_img=query_images,
                context_images=context_images if context_images else None
            )
            generator_type = "vlm"

        else:
            generation_result = self.generator.generate_response(
                question, 
                context_documents=retrieval_result["documents"]
            )
            generator_type = "text-only"
        
        results = {
            "question": question,
            "answer": generation_result["answer"],
            "retrieval_mode": self.config.retrieval_mode,
            "source_documents": retrieval_result["documents"],
            "retrieval_metadata": {
                "documents_retrieved": len(retrieval_result["documents"]),
                "enhancement_used": use_enhancement,
                "enhanced_query": retrieval_result.get("enhanced_query", question),
                "has_images_in_retrieved": retrieved_has_images,
                "images_retrieved": len(context_images),
                "query_images_provided": len(query_images) if query_images else 0
            },
            "generation_metadata": {
                **generation_result,
                "generator_type": generator_type,
                "images_used": len(context_images) if context_images else 0
            }
        }

        if self.config.retrieval_mode == "graph":
            results["graph_metadata"] = {
                "question_type": retrieval_result.get("question_type", "unknown"),
                "communities_used": len(retrieval_result.get("metadata", {}).get("communities", [])),
                "entities_found": retrieval_result.get("metadata", {}).get("total_entities_found", 0)
            }

        return results
    
    def get_image_data_urls(self, image_ids: List[str]) -> List[str]:
        """
        Gets image data URLs for a list of image IDs.
        
        args:
        - image_ids (List[str]): list of image IDs
        
        returns:
        - a list of image data URLs
        """
        if not self.image_store:
            return []
        
        data_urls = []
        for image_id in image_ids:
            data_url = self.image_store.get_image_data_url(image_id)
            if data_url:
                data_urls.append(data_url)
        
        return data_urls
    
    def search_images(self, query: str) -> List[Dict]:
        """
        Searches for images by caption or context.
        
        args:
        - query (str): search query
        
        returns:
        - a list of image metadata dictionaries
        """
        if not self.image_store:
            print("Image store not initialized. Set extract_images=True in config.")

            return []
        
        return self.image_store.search_images_by_caption(query)
    
    def get_document_images(self, document_name: str) -> List[Dict]:
        """
        Gets all images from a specific document.
        
        args:
        - document_name (str): name of the document
        
        returns:
        - list of image metadata dictionaries
        """
        if not self.image_store:
            print("Image store not initialized. Set extract_images=True in config.")

            return []
        
        return self.image_store.get_images_by_document(document_name)
    
    def get_image_store_stats(self) -> Dict:
        """
        Gets image store statistics.
        
        returns:
        - a dictionary with image store statistics
        """
        if not self.image_store:
            return {"error": "Image store not initialized"}
        
        return self.image_store.get_stats()
    
    def _extract_images_from_retrieved_chunks(self, retrieved_chunks: List) -> List[str]:
        """
        Extracts image data URLs from retrieved chunks.
        
        args:
        - retrieved_chunks (List): list of retrieved documents/chunks
        
        returns:
        - list of image data URLs
        """
        image_data_urls = []
        
        for chunk in retrieved_chunks:
            # check if chunk has metadata with image information
            if hasattr(chunk, 'metadata'):
                metadata = chunk.metadata

                if metadata.get("has_images", False) and "images" in metadata:
                    # get the image data URL
                    for img_info in metadata["images"]:
                        # temporary placeholder
                        pass
        
        return image_data_urls


if __name__ == "__main__":
    config = RAGConfig()
    rag_system = RAGSystem(config=config)
    
    # ingest documents once only when first processing documents
    # rag_system.ingest_documents("./documents/")
    
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        use_vlm = True
        if any(keyword in question.lower() for keyword in ['image', 'picture', 'photo', 'chart', 'diagram', 'graph', 'visual']):
            use_vlm_response = input("This question seems to be about images. Use VLM for better answers? (y/n): ")
            use_vlm = use_vlm_response.lower() == 'y'
        
        result = rag_system.query(question, use_vlm=use_vlm)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources retrieved: {result['retrieval_metadata']['documents_retrieved']}")
        for i in range(len(result['source_documents'])):
            print(f"Source {i}: {result['source_documents']}")

        if result['retrieval_metadata']['has_images_in_retrieved']:
            print(f"Note: Retrieved documents contain images")
            print(f"Images in context: {result['retrieval_metadata']['images_retrieved']}")
        
        print(f"Generator used: {result['generation_metadata']['generator_type']}")

        if result['retrieval_mode'] == "graph":
            if "graph_metadata" in result:
                print(f"Question type: {result['graph_metadata']['question_type']}")
                print(f"Communities used: {result['graph_metadata']['communities_used']}")

        if result['retrieval_metadata']['enhancement_used']:
            print(f"Enhanced query: {result['retrieval_metadata']['enhanced_query']}")