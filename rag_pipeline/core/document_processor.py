import os
import PyPDF2
import docx
from typing import List, Union, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import RAGConfig

class DocumentProcessor:
    """Handles document loading and text chunking with multi-format support"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: str) -> str:
        """Load text from various document formats"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext == '.docx':
            return self._load_docx(file_path)
        elif ext == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _load_docx(self, file_path: str) -> str:
        """Extract text from Word documents"""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _load_txt(self, file_path: str) -> str:
        """Load text from plain text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str, source_file: str = None) -> Tuple[List[str], List[dict]]:
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)

        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": i,
                "source_file": source_file or "unknown",
                "content_length": len(chunk),
                "total_chunks": len(chunks)
            }
            metadatas.append(metadata)

        return chunks, metadatas
    
    def process_directory(self, directory_path: str) -> Tuple[List[str], List[dict]]:
        """Process all supported documents in a directory"""
        all_chunks = []
        all_metadatas = []
        supported_extensions = {'.pdf', '.docx', '.txt'}
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in supported_extensions:
                try:
                    text = self.load_document(file_path)
                    chunks, metadatas = self.chunk_text(text, source_file=filename)
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
                    print(f"Processed {filename}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return all_chunks, all_metadatas