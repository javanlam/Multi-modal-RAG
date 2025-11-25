import os
from config.settings import RAGConfig
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.retriever import HyDERetriever
from core.generator import ResponseGenerator

class RAGSystem:
    """Main RAG system class that orchestrates all components"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig.from_env()
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store = VectorStoreManager(self.config)
        self.retriever = HyDERetriever(self.vector_store, self.config)
        self.generator = ResponseGenerator(self.config)
    
    def ingest_documents(self, source_path: str):
        """Ingest documents from a file or directory"""
        if os.path.isfile(source_path):
            # Single file processing
            text = self.document_processor.load_document(source_path)
            chunks, metadatas = self.document_processor.chunk_text(text, source_file=os.path.basename(source_path))
        else:
            # Directory processing
            chunks, metadatas = self.document_processor.process_directory(source_path)
        
        self.vector_store.add_documents(chunks, metadatas)
        print(f"Ingested {len(chunks)} document chunks")
    
    def query(self, question: str, use_enhancement: bool = True) -> dict:
        """Process a query and return response"""
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
    
    # Ingest documents once only
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