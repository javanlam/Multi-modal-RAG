import os
from typing import Dict
from config.settings import RAGConfig
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.retriever import HyDERetriever
from core.generator import ResponseGenerator
from models.embeddings import EmbeddingModel


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
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store = VectorStoreManager(self.config)
        self.retriever = HyDERetriever(self.vector_store, self.config)
        self.generator = ResponseGenerator(self.config)
        self.embedding = EmbeddingModel(self.config)
    
    def ingest_documents(self, source_path: str):
        """
        Ingests documents from a file or directory.
        
        args:
        - source_path (str): path of file or directory to process
        """
        if os.path.isfile(source_path):         # processes one file
            text = self.document_processor.load_document(source_path)
            chunks, metadatas = self.document_processor.chunk_text(text, source_file=os.path.basename(source_path))
        else:                                   # processes a directory
            chunks, metadatas = self.document_processor.process_directory(source_path)

        embeddings = self.embedding.encode(chunks)
        
        self.vector_store.add_documents(chunks, embeddings, metadatas)
        print(f"Ingested {len(chunks)} document chunks")

    def query(self, question: str, use_enhancement: bool = True) -> Dict:
        """
        Processes a query and returns the generated response.

        args:
        - question (str): user query to retrieve documents for
        - use_enhancement (bool): whether to enhance user prompt

        returns:
        - a dictionary containing the question, the generated answer, source documents, and additional information
        """
        retrieval_result = self.retriever.retrieve(question, use_enhancement)
        
        generation_result = self.generator.generate_response(
            question, 
            retrieval_result["documents"]
        )
        
        return {
            "question": question,
            "answer": generation_result["answer"],
            "source_documents": retrieval_result["documents"],
            "retrieval_metadata": {
                "documents_retrieved": len(retrieval_result["documents"]),
                "enhancement_used": use_enhancement,
                "enhanced_query": retrieval_result.get("enhanced_query", question)
            },
            "generation_metadata": generation_result
        }


if __name__ == "__main__":
    rag_system = RAGSystem()
    
    # ingest documents once only when first processing documents
    # rag_system.ingest_documents("./documents/")
    
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        result = rag_system.query(question)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources retrieved: {result['retrieval_metadata']['documents_retrieved']}")
        
        if result['retrieval_metadata']['enhancement_used']:
            print(f"Enhanced query: {result['retrieval_metadata']['enhanced_query']}")