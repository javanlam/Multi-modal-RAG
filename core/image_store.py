import json
import os
import base64
from typing import List, Dict, Optional
import hashlib
from datetime import datetime


class ImageStore:
    """
    Stores and manages extracted images from documents.
    """
    
    def __init__(self, storage_dir: str = "./image_store"):
        """
        Initializes the image store.
        
        args:
        - storage_dir (str): directory to store images and metadata
        """
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.images_dir = os.path.join(storage_dir, "images")
        
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """
        Loads metadata from file.
        
        returns:
        - a dictionary containing metadata
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {self.metadata_file}. Creating new metadata.")
        
        return {
            "images": {},
            "documents": {}, 
            "stats": {"total_images": 0}
        }
    
    def _save_metadata(self):
        """
        Saves metadata to file.
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def store_image(self, image_data_url: str, metadata: Dict) -> str:
        """
        Stores an image and its metadata.
        
        args:
        - image_data_url (str): data URL of the image
        - metadata (Dict): image metadata
        
        returns:
        - image ID for reference
        """
        if not image_data_url.startswith("data:image/"):
            print("Warning: Not a valid image data URL")
            return ""
        
        try:
            image_id = metadata.get("image_id")
            if not image_id:
                image_hash = hashlib.md5(image_data_url.encode()).hexdigest()
                image_id = f"{metadata.get('source_file', 'unknown')}_{metadata.get('page_num', 0)}_{image_hash[:8]}"
            
            # check if image already exists
            if image_id in self.metadata["images"]:
                print(f"Image {image_id} already exists in store")
                return image_id
            
            # extract image data from data URL
            header, data = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(data)
            
            # determine file extension from mime type
            mime_type = header.split(";")[0].split(":")[1]
            if "jpeg" in mime_type or "jpg" in mime_type:
                ext = "jpg"
            elif "png" in mime_type:
                ext = "png"
            else:
                ext = "bin"
            
            # save image file
            image_filename = f"{image_id}.{ext}"
            image_path = os.path.join(self.images_dir, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # create image record
            image_record = {
                "id": image_id,
                "filename": image_filename,
                "path": image_path,
                "source_file": metadata.get("source_file", "unknown"),
                "page_num": metadata.get("page_num", 0),
                "img_index": metadata.get("img_index", 0),
                "caption": metadata.get("caption", ""),
                "has_caption": metadata.get("has_caption", False),
                "context": metadata.get("context", "")[:500],
                "mime_type": mime_type,
                "stored_at": datetime.now().isoformat(),
                "size_bytes": len(image_bytes),
                "data_url": image_data_url  # original data URL for quick access
            }
            
            self.metadata["images"][image_id] = image_record
            
            doc_name = metadata.get("source_file", "unknown")
            if doc_name not in self.metadata["documents"]:
                self.metadata["documents"][doc_name] = []
            
            if image_id not in self.metadata["documents"][doc_name]:
                self.metadata["documents"][doc_name].append(image_id)
            
            self.metadata["stats"]["total_images"] = len(self.metadata["images"])
            
            self._save_metadata()
            
            print(f"Stored image {image_id} ({len(image_bytes)} bytes)")
            return image_id
            
        except Exception as e:
            print(f"Error storing image: {str(e)}")
            return None
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """
        Retrieves an image and its metadata.
        
        args:
        - image_id (str): ID of the image
        
        returns:
        - image metadata dictionary; OR None
        """
        return self.metadata["images"].get(image_id)
    
    def get_image_data_url(self, image_id: str) -> Optional[str]:
        """
        Retrieves an image data URL.
        
        args:
        - image_id (str): ID of the image
        
        returns:
        - image data URL; OR None
        """
        image = self.get_image(image_id)
        if image:
            if "data_url" in image:
                return image["data_url"]
            
            elif os.path.exists(image["path"]):
                # reconstruct data URL
                mime_type = image["mime_type"]

                with open(image["path"], 'rb') as f:
                    image_bytes = f.read()
                    base64_data = base64.b64encode(image_bytes).decode('utf-8')
                    return f"data:{mime_type};base64,{base64_data}"
                
        return None
    
    def get_images_by_document(self, document_name: str) -> List[Dict]:
        """
        Gets all images from a specific document.
        
        args:
        - document_name (str): name of the document
        
        returns:
        - a list of image metadata dictionaries
        """
        if document_name in self.metadata["documents"]:
            image_ids = self.metadata["documents"][document_name]

            return [self.get_image(img_id) for img_id in image_ids if self.get_image(img_id)]
        
        return []
    
    def get_images_by_ids(self, image_ids: List[str]) -> List[Dict]:
        """
        Gets images by a list of IDs.
        
        args:
        - image_ids (List[str]): list of image IDs
        
        returns:
        - a list of image metadata dictionaries
        """
        return [self.get_image(img_id) for img_id in image_ids if self.get_image(img_id)]
    
    def search_images_by_caption(self, query: str) -> List[Dict]:
        """
        Searches images by caption content.
        
        args:
        - query (str): search query
        
        returns:
        - a list of matching image metadata dictionaries
        """
        results = []
        query_lower = query.lower()
        
        for image_id, image in self.metadata["images"].items():
            caption = image.get("caption", "").lower()
            context = image.get("context", "").lower()
            
            if query_lower in caption or query_lower in context:
                results.append(image)
        
        return results
    
    def get_all_images(self) -> List[Dict]:
        """
        Gets all stored images.
        
        returns:
        - a list of all image metadata dictionaries
        """
        return list(self.metadata["images"].values())
    
    def clear(self):
        """
        Clears all stored images and metadata.
        """
        import shutil
        shutil.rmtree(self.images_dir, ignore_errors=True)
        self.metadata = {"images": {}, "documents": {}, "stats": {"total_images": 0}}
        self._save_metadata()
    
    def get_stats(self) -> Dict:
        """
        Gets storage statistics.
        
        returns:
        - a dictionary with statistics
        """
        return {
            "total_images": self.metadata["stats"]["total_images"],
            "documents_with_images": len(self.metadata["documents"]),
            "storage_dir": self.storage_dir
        }