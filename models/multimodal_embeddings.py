import torch
import base64
from io import BytesIO
from PIL import Image
from typing import List, Optional, Union
import numpy as np
import clip
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForTextEncoding
from LongCLIP.model import longclip
from config.settings import RAGConfig


class MultimodalEmbeddingModel:
    """
    Handles embedding for multimodal items using DINOv2 for images, and Talk2DINO for text.
    Embeddings are in DINOv2 space.
    """

    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the embedding model class with configurations provided.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load DINOv2 model
        self.image_processor = AutoImageProcessor.from_pretrained(config.dinov2_model_name, trust_remote_code=True, use_fast=True)
        self.image_model = AutoModel.from_pretrained(config.dinov2_model_name, trust_remote_code=True).to(self.device)
        self.image_model.eval()

        # load LongCLIP model
        self.longclip_model, self.longclip_preprocess = longclip.load("./LongCLIP/checkpoints/longclip-B.pt", device=self.device)

        # load Talk2DINO model
        self.clip_model, self.clip_preprocess = clip.load(config.clip_model_id, device=self.device, jit=False)
        self.text_model = AutoModel.from_pretrained(config.talk2dino_model_id, trust_remote_code=True).to(self.device)
        self.text_model.eval()

        self.embedding_dim = config.dinov2_embedding_dim

    def _data_url_to_image(self, data_url: str) -> Image.Image:
        """
        Converts a data URL to a PIL Image.

        args:
        - data_url (str): image URL

        returns:
        - the image represented as a PIL Image object
        """
        if not data_url.startswith("data:image/"):
            raise ValueError("Invalid image data URL")
        
        header, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        return Image.open(BytesIO(image_bytes)).convert("RGB")

    @torch.no_grad()
    def encode_image(self, image_data_url: str) -> List[float]:
        """
        Encodes a single image as a data URL into a DINO embedding vector.
        
        args:
        - image_data_url (str): an image data URL

        returns:
        - the corresponding DINO embedding vector
        """
        image = self._data_url_to_image(image_data_url)

        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.image_model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]  # (1, D)

        return embeddings.cpu().numpy().flatten().tolist()

    @torch.no_grad()
    def encode_text(self, text: str) -> List[float]:
        """
        Encodes text into DINOv2 embedding space using Talk2DINO.

        args:
        - text (str): the text to embed

        returns:
        - the corresponding embedding vector in DINOv2 embedding space
        """
        text_tokenized = longclip.tokenize(text).to(self.device)
        text_features = self.longclip_model.encode_text(text_tokenized)

        outputs = self.text_model.proj.project_clip_txt(text_features)

        return outputs.cpu().numpy().flatten().tolist()

    @torch.no_grad()
    def encode_batch_images(self, image_data_urls: List[str]) -> List[List[float]]:
        """
        Encodes multiple images as data URLs into DINO embedding vectors.
        
        args:
        - image_data_urls (List[str]): a list of multiple image data URLs

        returns:
        - a list of embeddings for each image
        """
        images = [self._data_url_to_image(url) for url in image_data_urls]
        
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.image_model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]  # (B, D)

        return embeddings.cpu().numpy().tolist()

    @torch.no_grad()
    def encode_batch_text(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a batch of text into DINOv2 embedding space using Talk2DINO.
        
        args:
        - texts (List[str]): a lst of texts to embed

        returns:
        - the corresponding embedding vector in DINOv2 embedding space
        """
        texts_tokenized = longclip.tokenize(texts).to(self.device)
        texts_features = self.longclip_model.encode_text(texts_tokenized)

        outputs = self.text_model.clip2dino_proj.project_clip_txt(texts_features)

        return outputs.cpu().numpy().tolist()

    def encode_query(self, text: Optional[str] = None, image_data_url: Optional[str] = None) -> List[float]:
        """
        Encodes a user query that may contain text and/or an image.

        args:
        - text (Optional[str]): possible text in user query
        - image_data_url (Optional[str]): possible image in user query as an image data URL

        returns:
        - a vector embedding of the received query item
        """
        if image_data_url:          # image
            return self.encode_image(image_data_url)
        
        elif text:                  # text
            return self.encode_text(text)
        
        else:                       # no image or text
            raise ValueError("Either text or image must be provided")