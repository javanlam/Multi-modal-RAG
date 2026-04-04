import os
import PyPDF2
import fitz
import base64
import docx
import json
from typing import List, Union, Tuple, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import RAGConfig
from .generator import ResponseGenerator
from .image_store import ImageStore


class DocumentProcessor:
    """
    Handles document loading and text chunking with multi-format support.
    """
    
    def __init__(
            self, 
            config: RAGConfig, 
            generator: Optional[ResponseGenerator] = None,
            image_store: Optional[ImageStore] = None
        ):
        """
        Initializes an instance of the document processor class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        - generator (Optional[ResponseGenerator]): optional generator for image captioning
        - image_store (Optional[ImageStore]): optional image store for managing extracted images
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.generator = generator
        self.image_store = image_store

    def load_document(self, file_path: str, extract_images: bool = True) -> Tuple[str, List[Dict]]:
        """
        Loads text from document at provided path.

        args:
        - file_path (str): path of document to load
        - extract_images (bool): whether to extract and process images

        returns:
        - a tuple of text extracted from the document and a list of image metadatas
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._load_pdf(file_path, extract_images)
        elif ext == '.docx':
            return self._load_docx(file_path, extract_images)
        elif ext == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_pdf(self, file_path: str, extract_images: bool = True) -> Tuple[str, List[Dict]]:
        """
        Extracts text from PDF files.
        
        args:
        - file_path (str): path of document to load
        - extract_images (bool): whether to extract and process images

        returns:
        - a tuple of text extracted from the document and a list of image metadatas
        """
        images_metadata = []
        text_parts = []

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_doc = fitz.open(file_path)

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text() + "\n"
                text_parts.append(page_text)

                if extract_images:
                    images_metadata.extend(self._extract_images_from_pdf_page(pdf_doc, page_num, page_text, file_path))

            pdf_doc.close()

        text = "\n".join(text_parts)

        return text, images_metadata
    
    def _extract_images_from_pdf_page(
            self, 
            pdf_doc: fitz.Document, 
            page_num: int, 
            page_text: str, 
            file_path: str
        ) -> List[Dict]:
        """
        Extracts images from a PDF page and generates captions.

        args:
        - pdf_doc (fitz.Document): the PDF file to process
        - page_num (int): page number to extract images from
        - page_text (str): text content of the page
        - file_path (str): path of document to process

        returns:
        - a list of dictionaries containing image metadata
        """
        images_metadata = []
        page = pdf_doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(pdf_doc, xref)

                # only process RGB or grayscale images
                if pix.n - pix.alpha < 4:
                    image_bytes = pix.tobytes()
                    
                    image_format = "png"
                    mime_type = "image/png"
                    
                    # convert to base64 data URL
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    image_data_url = f"data:{mime_type};base64,{base64_image}"
                else:
                    # skip other formats (not RGB or grayscale)
                    continue

                pix = None

                # get surrounding text context
                context = self._get_image_context(page, img)
                
                # generate caption
                caption = None
                if self.generator:
                    caption = self._generate_image_caption(image_data_url, context)
                
                image_metadata = {
                    "page_num": page_num + 1,
                    "img_index": img_index,
                    "context": context,
                    "image_data_url": image_data_url,
                    "caption": caption or f"Image from page {page_num + 1}",
                    "mime_type": mime_type,
                    "source_file": os.path.basename(file_path),
                    "position_in_text": len(page_text.split()),
                    "has_caption": caption is not None,
                    "stored": False
                }

                if self.image_store:
                    try:
                        stored_id = self.image_store.store_image(image_data_url, image_metadata)
                        if stored_id:
                            image_metadata["stored"] = True
                            image_metadata["image_id"] = stored_id
                    except Exception as e:
                        print(f"Error storing image {img_index} from page {page_num}: {str(e)}")

                images_metadata.append(image_metadata)
                
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {str(e)}")
                continue

        return images_metadata
    
    def _get_image_context(self, page: fitz.Page, img: Tuple, context_radius: int = 200) -> str:
        """
        Extracts text context around an image.

        args:
        - page (fitz.Page): the page object to obtain context from
        - img (Tuple): image tuple obtained from page.get_images()
        - context_radius (int): pixel radius to search for text around image

        returns:
        - a string containing context around the image
        """
        img_rect = page.get_image_bbox(img)

        # expand rectangle to get surrounding text
        expanded_rect = fitz.Rect(
            img_rect.x0 - context_radius,
            img_rect.y0 - context_radius,
            img_rect.x1 + context_radius,
            img_rect.y1 + context_radius
        )
        
        # extract text
        context = page.get_text("text", clip=expanded_rect)

        return context.strip()
    
    def _generate_image_caption(self, image_data_url: str, context: str) -> Optional[str]:
        """
        Generates caption for an image using a VLM.
        
        args:
        - image_data_url (str): encoded image data URL of the image to generate a caption for
        - context (str): surrounding text context from source document
        
        returns:
        - generated caption; OR None if failed
        """
        if not self.generator:
            return None
        
        try:
            prompt = f"""Based on the surrounding text context and the image content, provide a concise, descriptive caption for this image.

Surrounding context:
{context}

Your task is to produce a caption for the image with context from the original source.
Describe what you see in the image, and give a comprehensive caption that explains the image content as well as the context of the document.
However, do not give information that is too excessive in the caption. 
Remain factual in writing the caption.

The caption should:
1. Describe the visual content clearly
2. Incorporate relevant information from the context
3. Be concise (1-2 sentences)
4. Start with "Image: "

Caption:"""

            response = self.generator.llm.generate_response(
                user_prompt=prompt,
                images=[image_data_url]
            )
            
            if "error" not in response:
                caption = response.get("answer", "").strip()

                if caption.startswith("Caption:"):
                    caption = caption[8:].strip()

                return caption
        
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
        
        return None

    def _load_docx(self, file_path: str, extract_images: bool = True) -> Tuple[str, List[Dict]]:
        """
        Extracts text from Word documents.

        args:
        - file_path (str): path of document to load
        - extract_images (bool): whether to extract and process images

        returns:
        - a tuple of text extracted from the document and a list of image metadatas
        """
        doc = docx.Document(file_path)

        text_parts = []
        images_metadata = []

        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)

        text = "\n".join(text_parts)

        return text, images_metadata

    def _load_txt(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Loads text from plain text files.
        
        args:
        - file_path (str): path of document to load

        returns:
        - a tuple of text extracted from the document and a list of image metadatas
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(), []

    def insert_image_captions(self, text: str, images_metadata: List[Dict]) -> str:
        """
        Inserts image captions into the text at appropriate positions.
        
        args:
        - text (str): original text
        - images_metadata (List[dict]): a list of image metadata dictionaries
        
        returns:
        - text with image captions inserted
        """
        if not images_metadata:
            return text
        
        # group captions by page number
        pages_captions = {}
        for image in images_metadata:
            page_num = int(image.get("page_num", 1))
            caption = image.get("caption", "")
            if caption:
                pages_captions.setdefault(page_num, []).append(f"[IMAGE: {caption}]")

        if not pages_captions:
            return text

        max_page = max(pages_captions.keys())

        # split text into pages
        if '\f' in text:
            pages = text.split('\f')
            separator = '\f'

        else:
            lines = text.splitlines()
            lines_per_page = max(1, -(-len(lines) // max_page))
            pages = []

            for i in range(max_page):
                start = i * lines_per_page
                end = start + lines_per_page
                pages.append("\n".join(lines[start:end]))

            # append leftover lines to last page
            if end < len(lines):
                pages[-1] = pages[-1] + ("\n" if pages[-1] else "") + "\n".join(lines[end:])

            separator = "\n\n"

        # ensure at least max_page pages
        while len(pages) < max_page:
            pages.append("")

        # append captions to end of each page
        for page_index in range(1, max_page + 1):
            caps = pages_captions.get(page_index)
            if caps:
                addition = "\n" + "\n".join(caps)
                pages[page_index - 1] = (pages[page_index - 1] + addition).rstrip()

        new_text = separator.join(pages)
        return new_text

    def chunk_text(self, text: str, source_file: str = None, images_metadata: List[Dict] = None) -> Tuple[List[str], List[Dict]]:
        """
        Splits text into chunks for easier processing.
        
        args:
        - text (str): the text to perform chunking on
        - source_file (str): path to source file of the text
        - images_metadata (List[dict]): a list of image metadata dictionaries

        returns:
        - a tuple containing a list of chunked text pieces, and a list of dictionaries containing chunk metadata
        """
        chunks = self.text_splitter.split_text(text)

        metadatas = []

        for i, chunk in enumerate(chunks):
            has_image = "[IMAGE:" in chunk if images_metadata else False

            metadata = {
                "chunk_index": i,
                "source_file": source_file or "unknown",
                "content_length": len(chunk),
                "total_chunks": len(chunks),
                "has_image": has_image,
                "image_count": chunk.count("[IMAGE:") if has_image else 0,
                "images": []
            }

            if images_metadata and has_image:
                for img_meta in images_metadata:
                    img_caption = img_meta.get("caption", "")

                    if img_caption and img_caption in chunk:
                        metadata["images"].append({
                            "caption": img_caption,
                            "page": img_meta.get("page_num"),
                            "has_caption": img_meta.get("has_caption", False)
                        })

            metadata["images"] = json.dumps(metadata["images"])     # lists cannot be stored in ChromaDB

            metadatas.append(metadata)

        return chunks, metadatas

    def process_directory(self, directory_path: str, extract_images: bool = True) -> Tuple[List[str], List[dict], List[dict]]:
        """
        Processes all supported documents in a directory.
        
        args:
        - directory_path (str): path of directory to process
        - extract_images (bool): whether to extract and process images

        returns:
        - a tuple containing a list of chunked text pieces, a list of dictionaries containing chunk metadata, 
            and a list of dictionaries containing image metadata
        """
        all_chunks = []
        all_metadatas = []
        image_metadata_list = []
        supported_extensions = {'.pdf', '.docx', '.txt'}
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in supported_extensions:
                try:
                    text, images_metadata = self.load_document(file_path, extract_images)
                    text_captioned = self.insert_image_captions(text, images_metadata)
                    image_metadata_list.extend(images_metadata)

                    chunks, metadatas = self.chunk_text(text_captioned, source_file=filename, images_metadata=images_metadata)
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)

                    print(f"Processed {filename}: {len(chunks)} chunks")
                    print(f"{len(images_metadata)} images extracted")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return all_chunks, all_metadatas, image_metadata_list