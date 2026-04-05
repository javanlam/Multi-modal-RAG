import base64
from mimetypes import guess_type
import fitz
import os
from PIL import Image
from io import BytesIO
from typing import List, Dict, Tuple


def local_image_to_data_url(image_path: str) -> str:
    """
    Helper function to encode a local image into data URL.
    Retrieved from https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision?tabs=rest

    args:
    - image_path (str): path to image in local file system

    returns:
    - the encoded data URL
    """
    # guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # default MIME type if none is found

    # read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def data_url_to_bytes(data_url: str) -> Tuple[bytes, str]:
    """
    Converts an image data URL to a byte array.

    args:
    - data_url (str): the image data URL

    returns:
    - a tuple containing the byte array of the image and the MIME type
    """
    if not data_url.startswith("data:image/"):
        raise ValueError("Invalid image data URL")
    
    header, encoded = data_url.split(",", 1)
    mime_type = header.split(":")[1].split(";")[0]
    
    image_bytes = base64.b64decode(encoded)

    return image_bytes, mime_type


def data_url_to_pil(data_url: str) -> Image.Image:
    """
    Converts an image data URL to a PIL Image.
    
    args:
    - data_url (str): the image data URL

    returns:
    - a PIL Image object of the image
    """
    if not data_url.startswith("data:image/"):
        raise ValueError("Invalid image data URL")
    
    header, encoded = data_url.split(",", 1)
    
    image_bytes = base64.b64decode(encoded)

    return Image.open(BytesIO(image_bytes)).convert("RGB")