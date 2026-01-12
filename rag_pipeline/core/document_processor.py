import os
import PyPDF2
import docx
from typing import List, Union, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import RAGConfig


class DocumentProcessor:
    """
    Handles document loading and text chunking with multi-format support.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the document processor class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_document(self, file_path: str) -> str:
        """
        Loads text from document at provided path.

        args:
        - file_path (str): path of document to load

        returns:
        - text extracted from the document; OR an error message
        """
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
        """
        Extracts text from PDF files.
        
        args:
        - file_path (str): path of document to load

        returns:
        - text extracted from the document; OR an error message
        """
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _load_docx(self, file_path: str) -> str:
        """
        Extracts text from Word documents.

        args:
        - file_path (str): path of document to load

        returns:
        - text extracted from the document; OR an error message
        """
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _load_txt(self, file_path: str) -> str:
        """
        Loads text from plain text files.
        
        args:
        - file_path (str): path of document to load

        returns:
        - text extracted from the document; OR an error message
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()    

    def chunk_text(self, text: str, source_file: str = None) -> Tuple[List[str], List[dict]]:
        """
        Splits text into chunks for easier processing.
        
        args:
        - text (str): the text to perform chunking on
        - source_file (str): path to source file of the text

        returns:
        - a tuple containing a list of chunked text pieces, and a list of dictionaries containing chunk metadata
        """
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
        """
        Processes all supported documents in a directory.
        
        args:
        - directory_path (str): path of directory to process

        returns:
        - a tuple containing a list of chunked text pieces, and a list of dictionaries containing chunk metadata
        """
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