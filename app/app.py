import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from config.settings import RAGConfig
from RAGPipeline import RAGSystem


class QueryRequest(BaseModel):
    question: str
    query_images: Optional[List[str]] = None
    use_enhancement: bool = True
    use_vlm: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieval_mode: str
    source_documents: List[Any]
    context_images: List[str]
    retrieved_image_ids: List[str]
    retrieval_metadata: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    graph_metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: Optional[int] = None
    images_stored: Optional[int] = None


rag_system: Optional[RAGSystem] = None
config: Optional[RAGConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global rag_system
    global config

    config = RAGConfig.from_env()
    rag_system = RAGSystem(config=config)

    print("RAGSystem initialised with configuration:")
    print(f" - Retrieval mode: {config.retrieval_mode}")
    print(f" - Multimodal: {config.use_multimodal}")
    print(f" - Image extraction: {config.extract_images}")
    
    yield
    
    # shutdown
    print("Shutting down RAG system...")


app = FastAPI(
    title="RAG System",
    description="Multi-modal RAG",
    version="1.0.0",
    lifespan=lifespan
)


origins = [
    "http://localhost:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Ingests a single document into the RAG system.

    args:
    - file (UploadFile): the uploaded file

    returns:
    - an IngestResponse object returning the ingestion outcome
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialised")

    # save to a temporary location
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, rag_system.ingest_documents, tmp_path)

        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            chunks_ingested=None,
            images_stored=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # clean up temporary file
        os.unlink(tmp_path)


def make_json_serializable(obj: Any):
    """
    Converts an object into a JSON-serializable data type.

    args:
    - obj (Any): the object to convert

    returns:
    - a JSON-serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}

    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):
        return {
            "text": obj.page_content,
            "metadata": make_json_serializable(obj.metadata)
        }
    
    if hasattr(obj, "dict") and callable(obj.dict):
        return make_json_serializable(obj.dict())
    
    if hasattr(obj, "__dict__"):
        return make_json_serializable(obj.__dict__)
    
    # fallback by string conversion
    return str(obj)


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Processes a query and returns the generated answer.

    args:
    - request (QueryRequest): a query request containing the query text and images

    returns:
    - a QueryResponse object containing the answer and response metadata
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialised")

    import asyncio
    loop = asyncio.get_running_loop()

    try:
        result: Dict = await loop.run_in_executor(
            None,
            rag_system.query,
            request.question,
            request.query_images,
            request.use_enhancement,
            request.use_vlm
        )
        serializable_result = make_json_serializable(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    source_docs = []
    for doc in serializable_result.get("source_documents", []):
        if isinstance(doc, str):
            # plain text
            source_docs.append({"text": doc, "metadata": {}})
        elif isinstance(doc, dict):
            if "text" not in doc and "page_content" in doc:
                doc["text"] = doc["page_content"]
            if "text" not in doc:
                doc["text"] = str(doc)
            if "metadata" not in doc:
                doc["metadata"] = {}
            source_docs.append(doc)
        else:
            # fallback by string conversion
            source_docs.append({"text": str(doc), "metadata": {}})

    response = QueryResponse(
        question=serializable_result["question"],
        answer=serializable_result["answer"],
        retrieval_mode=serializable_result["retrieval_mode"],
        source_documents=source_docs,
        context_images=serializable_result.get("context_images", []),
        retrieved_image_ids=serializable_result.get("retrieved_image_ids", []),
        retrieval_metadata=serializable_result["retrieval_metadata"],
        generation_metadata=serializable_result["generation_metadata"],
        graph_metadata=serializable_result.get("graph_metadata")
    )
    
    return response


@app.get("/health")
async def health_check():
    """
    Simple health check.
    """
    if rag_system is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    
    return {"status": "ok", "retrieval_mode": rag_system.config.retrieval_mode}