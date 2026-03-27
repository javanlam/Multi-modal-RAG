import base64
from mimetypes import guess_type
import fitz
import os
from PIL import Image
from io import BytesIO
from typing import List, Dict


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


def extract_images_with_context(pdf_path: str, output_image_dir: str) -> List[Dict]:
    """
    Extracts images and their surrounding text context from a PDF document.

    Args:
        pdf_path (str): Path to the source PDF file
        output_image_dir (str): Directory where extracted images will be saved
    """

    os.makedirs(output_image_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    image_paths_and_contexts = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # all text blocks from the current page (including coordinates)
        text_blocks = page.get_text("blocks")
        text_blocks.sort(key=lambda block: block[1])  # sort vertically

        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            # coordinates of current image
            img_rect = page.get_image_bbox(img)
            img_x0, img_y0, img_x1, img_y1 = img_rect
            img_width = img_x1 - img_x0
            img_height = img_y1 - img_y0

            # image location and size (for debugging purposes)
            # print(f"Page {page_num+1}, Image {img_index+1}:")
            # print(f"  Location: ({img_x0:.2f}, {img_y0:.2f})")
            # print(f"  Size: {img_width:.2f} x {img_height:.2f}")

            image_filename = f"page_{page_num+1}_img_{img_index+1}.png"
            image_path = os.path.join(output_image_dir, image_filename)
            pix.save(image_path)    # saves image to file system

            context_blocks = []
            for block in text_blocks:
                block_rect = fitz.Rect(block[:4])
                block_text = block[4].strip()

                # text around image in source document layout is used as context
                if (abs(block_rect.y1 - img_y0) < 50 or abs(block_rect.y0 - img_y1) < 50):
                    context_blocks.append(block_text)

            context_text = " ".join(context_blocks)

            image_paths_and_contexts.append({
                "context": context_text,
                "page_num": page_num,
                "img_index": img_index,
                "image_filename": image_filename,
                "image_path": image_path,
            })

            pix = None

    doc.close()

    return image_paths_and_contexts