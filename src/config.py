import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Database configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_multimodal")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgrespassword")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Embedding model (Multimodal CLIP ViT-B-32)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "clip-ViT-B-32")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Documents and outputs
DEFAULT_PDF_PATH = BASE_DIR / os.getenv("DEFAULT_PDF_PATH", "attention is all you need.pdf")
OUTPUT_IMAGE_DIR = BASE_DIR / os.getenv("OUTPUT_IMAGE_DIR", "extracted_images")

# Vector store table name for multimodal 512-dim CLIP vectors
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "pdf_multimodal_clip_nodes")

def get_available_ollama_models(ollama_base_url: str = "http://127.0.0.1:11434") -> list:
    """Retrieve the list of downloaded models in the local Ollama instance (instant check, no torch)."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request(f"{ollama_base_url}/api/tags", headers={"User-Agent": "AntigravityRAG"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

