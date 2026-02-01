from .main import RAGSystem
from config.settings import RAGConfig

# Initialize with custom config
config = RAGConfig(
    chunk_size=800,
    top_k=5,
    llm_provider="openai"
)

rag = RAGSystem(config)

# Ingest documents
rag.ingest_documents("path")

# Query the system
result = rag.query("What are the main challenges in AI implementation?")
print(result["answer"])