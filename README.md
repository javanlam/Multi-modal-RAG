# Multi-modal Retrieval-Augmented Generation (RAG)

This repo implements a multimodal Retrieval-Augmented Generation (RAG) pipeline using separate representations for text and visual content,
with [DINOv2](https://github.com/facebookresearch/dinov2) as visual features, and [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO)
for text-vision alignment.
Our pipeline uses mainly vector databases for storage of embeddings.
Text-to-text, text-to-image, and image-to-image query retrieval modes are supported.

## Features
- **Dual‑representation storage**: Text chunks and image features are indexed separately to preserve fine‑grained visual details.
- **Cross‑modal retrieval without SOTA foundation models**: Uses DINOv2 + Talk2DINO to align text and the visual feature space.
- **Multiple retrieval modes**: Text‑to‑text, text‑to‑image, and image‑to‑image.
- **End‑to‑end grounding**: Retrieved text and images are fed to a Multimodal LLM (MLLM) for answer generation.

## Architecture
1. Document Preprocessing (for pdf and docx documents): Text chunking, inline image extraction, MLLM captioning conditioned on document context.
2. Embedding: Text chunks with ```sentence-transformers/all-MiniLM-L6-v2```, images with DINOv2.
3. Storage: In a ChromaDB vector database.
4. Retrieval: By cosine similarity of vector embeddings.
5. Generation: Using raw query and retrieved text and visual context.

## Installation
Clone the repository:
```bash
git clone https://github.com/javanlam/Multi-modal-RAG.git
cd Multi-modal-RAG
```
Quick setup with our provided script: (ensure conda is installed, we use conda for environment setup)
```bash
chmod +x setup.sh   # allow execution of bash script
./setup.sh
```

See [```requirements.txt```](requirements.txt) for a detailed list of dependencies.

To use with a MLLM, you must set up your API keys in ```.env```. 
Create a new file named ```.env```, and paste your desired API keys as follows.

We support OpenAI, Azure OpenAI, Google Gemini, and Qwen.
```ini
# file: .env
OPENAI_API_KEY=<your api key for OpenAI or Azure OpenAI>
GEMINI_API_KEY=<your api key for Google Gemini>
DASHSCOPE_API_KEY=<your api key for Qwen>
```

## Example Code
Use the repo root as the working directory.
```python
from config.settings import RAGConfig
from RAGPipeline import RAGSystem

# check config/settings.py for default arguments
# only provide arguments as necessary
config = RAGConfig(
    llm_provider="openai-azure",
        # "openai" for OpenAI, "openai-azure" for Azure OpenAI, "google" for Google Gemini, "qwen" for Qwen
    llm_model="gpt-4o-mini",
        # the LLM model you want to use
    temperature=0.9,
        # LLM temperature
    persist_directory="./chroma_db",
        # directory to place vector database files
    collection_name="documents",
        # name of vector database
    extract_images=True,
        # extract images from documents
    generate_image_captions=True,
        # generate captions for extracted images using MLLM
    image_store_dir="./image_store",
        # directory where extracted images are stored
    image_embeddings_collection="image_embeddings",
        # name of vector database for image embeddings
    caption_embeddings_collection="caption_embeddings",
        # name of vector database for caption embeddings
    vlm_model="gpt-4o-mini",
        # MLLM to use as vision-language model
    image_caption_prompt="Provide a concise, descriptive caption for this image based on the surrounding text context.",
        # prompt for MLLM captioning
    top_k=3,
        # number of most similar items to retrieve
    similarity_threshold=0.7,
        # cosine similarity threshold for retrieval
)

rag_system = RAGSystem(config=config)

# ingest documents from directory "documents"
rag_system.ingest_documents(source_path="./documents")

results = rag_system.query(
    question="What is RAG?",        # query
    query_images=[],                # images for query, if any (must be image data URLs)
    use_enhancement=True,           # use the Hypothetical Document Embedding (HyDE) technique for retrieval
    use_vlm=True                    # include retrieved visual context for generation
)

# final generated answer
print(results["answer"])
```

## Example Notebooks
For all example notebooks inside ```example_notebooks/```, please copy them to the repo root before running code inside.

## Repository Structure
```
  .
  ├── Multi-modal-RAG/          # repo root
  │   ├── config/               # configuration scripts
  │   ├── core/                 # document processor, storage manager, generator
  │   ├── models/               # embedding models, LLM/MLLM wrappers
  │   ├── utils/                # helper functions
  │   ├── example_notebooks/    # Python notebooks for examples, tests, and benchmarking
  └── └── README.md             # this file
```

## Acknowledgements
This project builds on open‑source work:
- [DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI Research)
- [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO) (University of Modena and Reggio, ISTI-CNR, University of Pisa)
- [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)