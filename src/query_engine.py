import sys
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from PIL import Image
import psycopg2

# Ensure UTF-8 printing on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    PG_TABLE_NAME,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)
from src.embeddings import get_clip_manager

logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """
    Multimodal Retriever querying the PGVector database for text, tables, and images.
    Accepts EITHER text OR an image as the search query.
    Calculates cosine similarity in the shared 512-dim CLIP latent space.
    """

    def __init__(self, table_name: str = PG_TABLE_NAME, similarity_top_k: int = 4):
        self.table_name = table_name
        self.similarity_top_k = similarity_top_k
        self.clip_manager = get_clip_manager()

    def _query_pgvector_by_embedding(
        self, query_embedding: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Query PGVector directly using cosine similarity (1 - cosine distance <=>).
        """
        table = f"data_{self.table_name}"
        embedding_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
        
        results = []
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            cur = conn.cursor()
            
            # Use PGVector cosine distance <=> (1 - distance = cosine similarity)
            sql = f"""
                SELECT 
                    id,
                    text,
                    metadata_,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM {table}
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s;
            """
            cur.execute(sql, (embedding_str, embedding_str, top_k))
            rows = cur.fetchall()
            
            for row in rows:
                node_id, text, metadata, score = row
                meta = metadata or {}
                results.append({
                    "node_id": node_id,
                    "score": float(score) if score is not None else 0.0,
                    "type": meta.get("type", "unknown"),
                    "page": meta.get("page"),
                    "doc_name": meta.get("doc_name"),
                    "metadata": meta,
                    "text": text or "",
                    "image_path": meta.get("image_path"),
                    "caption": meta.get("caption"),
                })

            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error querying PGVector: {e}")
            raise e

        return results

    def retrieve_by_text(self, query_str: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k multimodal results using a text query."""
        k = top_k or self.similarity_top_k
        logger.info(f"Retrieving with text query: '{query_str}' (top_k={k})")
        text_emb = self.clip_manager.embed_text(query_str)
        return self._query_pgvector_by_embedding(text_emb, top_k=k)

    def retrieve_by_image(
        self, image_input: Union[str, Path, Image.Image], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k multimodal results using an image query."""
        k = top_k or self.similarity_top_k
        img_desc = str(image_input) if isinstance(image_input, (str, Path)) else "PIL Image"
        logger.info(f"Retrieving with image query: {img_desc} (top_k={k})")
        img_emb = self.clip_manager.embed_image(image_input)
        return self._query_pgvector_by_embedding(img_emb, top_k=k)

    def retrieve(
        self, query: Union[str, Path, Image.Image], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Unified retrieval function accepting either text or image query.
        Automatically detects input modality.
        """
        # If PIL Image instance
        if isinstance(query, Image.Image):
            return self.retrieve_by_image(query, top_k=top_k)

        # If string or Path, check if it points to an existing image file
        if isinstance(query, (str, Path)):
            query_path = Path(query)
            image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
            if query_path.exists() and query_path.suffix.lower() in image_extensions:
                return self.retrieve_by_image(query_path, top_k=top_k)
            else:
                return self.retrieve_by_text(str(query), top_k=top_k)

        raise ValueError(f"Unsupported query type: {type(query)}")

    def print_results(self, query: Any, results: List[Dict[str, Any]]):
        """Display readable formatted results for console and inspection."""
        query_desc = str(query) if not isinstance(query, Image.Image) else "PIL Image Query"
        print(f"\n{'='*75}")
        print(f"MULTIMODAL RESULTS FOR: '{query_desc}'")
        print(f"{'='*75}")
        if not results:
            print("No matching items found.")
            return

        for i, res in enumerate(results, start=1):
            node_type = res["type"].upper()
            page = res["page"]
            score = f"{res['score']:.4f}"
            print(f"\n--- [Result #{i}] Type: {node_type} | Page: {page} | Cosine Similarity: {score} ---")
            if res["type"] == "image":
                print(f" Image Path : {res.get('image_path')}")
                print(f" Caption    : {res.get('caption')}")
            content_preview = res['text'][:280].replace('\n', ' ')
            print(f" Content    : {content_preview}...")
        print(f"{'='*75}\n")
