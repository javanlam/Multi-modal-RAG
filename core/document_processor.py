import os
import fitz
import base64
import docx
import json
import subprocess
import tempfile
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
        doc = fitz.open(file_path)
        all_images_metadata = []
        page_texts = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # sorted items in reading order
            items = self._get_sorted_page_items(page, extract_images)
            
            # build page content
            page_text, page_images = self._build_page_content(doc, page, items, page_num, file_path)
            
            all_images_metadata.extend(page_images)
            page_texts.append(page_text)
            page_texts.append("\f")   # page separator

        doc.close()
        full_text = "\n".join(page_texts).rstrip("\f")
        return full_text, all_images_metadata

    def _get_sorted_page_items(self, page: fitz.Page, extract_images: bool) -> List[Dict]:
        """
        Extracts text and images from a page and sort by reading order.
        
        args:
        - page (fitz.Page): the document page to process
        - extract_images (bool): whether to extract and process images

        returns:
        - a list of dictionaries containing text and image items on the page
        """
        # text
        words = page.get_text("words")  # [x0, y0, x1, y1, word, block_no, line_no, word_no]
        text_items = [{"type": "text", "text": w[4], "y0": w[1], "x0": w[0]} for w in words]
        
        # images
        image_items = []

        if extract_images:
            for img_info in page.get_image_info():
                bbox = fitz.Rect(img_info["bbox"])
                image_items.append({
                    "type": "image",
                    "bbox": bbox,
                    "y0": bbox.y0,
                    "x0": bbox.x0,
                    "xref": img_info.get("xref", 0)
                })
        
        all_items = text_items + image_items
        all_items.sort(key=lambda item: (item["y0"], item["x0"]))

        return all_items

    def _build_page_content(
            self, 
            doc: fitz.Document, 
            page: fitz.Page, 
            items: List[Dict],
            page_num: int, 
            file_path: str
        ) -> Tuple[str, List[Dict]]:
        """
        Rebuilds content on a certain document page.
        
        args:
        - doc (fitz.Document): the document to process
        - page (fitz.Page): the document page to process
        - items (List[Dict]): the list of items on the page in reading order
        - page_num (int): page number of current page
        - file_path (str): path of document to process

        returns:
        - a tuple of the text on page and a list of dictionaries containing image metadata
        """
        lines = []
        current_line_words = []
        last_y0 = None
        threshold = 5  # pt
        page_images = []
        
        for item in items:
            if item["type"] == "text":
                y0 = item["y0"]

                if last_y0 is not None and abs(y0 - last_y0) > threshold:
                    # new line
                    if current_line_words:
                        lines.append(" ".join(current_line_words))
                        current_line_words = []

                current_line_words.append(item["text"])
                last_y0 = y0

            else:  # image
                if current_line_words:
                    lines.append(" ".join(current_line_words))
                    current_line_words = []

                # process image and add marker as its own line
                marker, metadata = self._process_image_item(doc, page, item, page_num, file_path)
                lines.append(marker)

                if metadata:
                    page_images.append(metadata)
        
        if current_line_words:
            lines.append(" ".join(current_line_words))
        
        return "\n".join(lines), page_images

    def _process_image_item(
            self, 
            doc: fitz.Document, 
            page: fitz.Page, 
            item: Dict,
            page_num: int, 
            file_path: str
        ) -> Tuple[str, Optional[Dict]]:
        """
        Processes image items on a certain document page.
        
        args:
        - doc (fitz.Document): the document to process
        - page (fitz.Page): the document page to process
        - items (List[Dict]): the list of items on the page in reading order
        - page_num (int): page number of current page
        - file_path (str): path of document to process

        returns:
        - a tuple of a string containing the processed caption and a dictionary containing image metadata
        """
        try:
            xref = item["xref"]
            bbox = item["bbox"]
            
            # extract pixmap
            if xref == 0:
                pix = page.get_pixmap(clip=bbox)
            else:
                pix = fitz.Pixmap(doc, xref)
            
            if pix.n - pix.alpha >= 4:
                pix = None
                return "[IMAGE: unsupported format]", None
            
            image_bytes = pix.tobytes()
            mime_type = "image/png"
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:{mime_type};base64,{base64_image}"
            pix = None
            
            context = self._get_image_context_from_rect(page, bbox)
            caption = None

            if self.generator:
                caption = self._generate_image_caption(image_data_url, context)
            
            caption_text = caption or f"Image from page {page_num + 1}"
            marker = f"[IMAGE: {caption_text}]"
            
            metadata = {
                "page_num": page_num + 1,
                "img_index": None,   # set later if needed
                "context": context,
                "image_data_url": image_data_url,
                "caption": caption_text,
                "mime_type": mime_type,
                "source_file": os.path.basename(file_path),
                "has_caption": caption is not None,
                "stored": False
            }
            
            if self.image_store:
                try:
                    stored_id = self.image_store.store_image(image_data_url, metadata)
                    if stored_id:
                        metadata["stored"] = True
                        metadata["image_id"] = stored_id
                except Exception as e:
                    print(f"Error storing image: {e}")
            
            return marker, metadata
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return "[IMAGE: extraction failed]", None
        
    def _get_image_context_from_rect(self, page: fitz.Page, img_rect: fitz.Rect, context_radius: int = 200) -> str:
        """
        Extracts text surrounding an image as context for captioning.

        args:
        - page (fitz.Page): the page object to obtain context from
        - img_rect (fitz.Rect): bounding box of the image to obtain context for
        - context_radius (int): pixel radius to search for text around image
        
        returns:
        - a string containing context around the image
        """
        expanded_rect = fitz.Rect(
            img_rect.x0 - context_radius,
            img_rect.y0 - context_radius,
            img_rect.x1 + context_radius,
            img_rect.y1 + context_radius
        )

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
5. NOT express that you based it off the provided context

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
        if not extract_images:
            doc = docx.Document(file_path)

            text_parts = []
            images_metadata = []

            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)

            text = "\n".join(text_parts)

            return text, images_metadata
        
        # convert to PDF and handle with _load_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            pdf_path = tmp_pdf.name

        try:
            subprocess.run([
                "libreoffice", "--headless", "--convert-to", "pdf",
                "--outdir", os.path.dirname(pdf_path), file_path
            ], check=True, capture_output=True)
            
            expected_pdf = os.path.splitext(file_path)[0] + ".pdf"

            # may be placed in outdir, check if file exists
            if not os.path.exists(expected_pdf):
                if os.path.exists(pdf_path):
                    # already at expected path
                    pass
                else:
                    raise FileNotFoundError("PDF conversion failed")
            else:
                # not at expected path
                os.replace(expected_pdf, pdf_path)

            text, images_metadata = self._load_pdf(pdf_path, extract_images=True)
            
            for img in images_metadata:
                img["source_file"] = os.path.basename(file_path)
            
            return text, images_metadata
        
        finally:
            # delete temporary PDF
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

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
                    image_metadata_list.extend(images_metadata)

                    chunks, metadatas = self.chunk_text(text, source_file=filename, images_metadata=images_metadata)
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)

                    print(f"Processed {filename}: {len(chunks)} chunks")
                    print(f"{len(images_metadata)} images extracted")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return all_chunks, all_metadatas, image_metadata_list