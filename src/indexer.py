import logging
from typing import Dict, Any, List
from pathlib import Path
import psycopg2
from llama_index.core import VectorStoreIndex, StorageContext
from src.config import (
    DEFAULT_PDF_PATH,
    OUTPUT_IMAGE_DIR,
    PG_TABLE_NAME,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)
from src.embeddings import get_embedding_model, get_clip_manager
from src.storage import get_vector_store
from src.extractor import MultimodalPDFExtractor

logger = logging.getLogger(__name__)

class MultimodalIndexer:
    """
    Orchestrates PDF extraction, multimodal CLIP embedding generation (joint text and image),
    document SHA-256 hash tracking, and storage into PostgreSQL with PGVector.
    """

    def __init__(
        self,
        pdf_path: str | Path = DEFAULT_PDF_PATH,
        output_image_dir: str | Path = OUTPUT_IMAGE_DIR,
        table_name: str = PG_TABLE_NAME,
    ):
        self.pdf_path = Path(pdf_path)
        self.output_image_dir = Path(output_image_dir)
        self.table_name = table_name
        self.clip_manager = get_clip_manager()

    def check_existing_doc_hash(self, doc_hash: str) -> int:
        """Check how many records already exist in PGVector with the same document hash."""
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            cur = conn.cursor()
            target_table = f"data_{self.table_name}"
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = '{target_table}'
                );
            """)
            table_exists = cur.fetchone()[0]
            if not table_exists:
                cur.close()
                conn.close()
                return 0

            cur.execute(f"""
                SELECT count(*) FROM {target_table}
                WHERE metadata_->>'doc_hash' = %s;
            """, (doc_hash,))
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"Unable to check existing doc_hash: {e}")
            return 0

    def delete_existing_doc(self, doc_hash: str):
        """Remove previously indexed records for this document hash to allow clean reindexing."""
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            cur = conn.cursor()
            target_table = f"data_{self.table_name}"
            cur.execute(f"""
                DELETE FROM {target_table}
                WHERE metadata_->>'doc_hash' = %s;
            """, (doc_hash,))
            conn.commit()
            deleted_count = cur.rowcount
            cur.close()
            conn.close()
            logger.info(f"Purged {deleted_count} existing records for doc_hash {doc_hash[:10]}... from {target_table}")
        except Exception as e:
            logger.warning(f"Error purging existing doc records: {e}")

    def run(self, force_reindex: bool = True) -> Dict[str, Any]:
        """
        Execute complete extraction, joint multimodal CLIP embedding, and PGVector indexation.
        """
        logger.info(f"=== Starting Multimodal CLIP Indexing Pipeline for {self.pdf_path.name} ===")

        # 1. Extract Text, Tables, and Images from PDF
        extractor = MultimodalPDFExtractor(
            pdf_path=self.pdf_path,
            output_image_dir=self.output_image_dir,
        )
        nodes, images_meta = extractor.extract_all_nodes()
        doc_hash = extractor.doc_hash

        logger.info(f"Document SHA-256: {doc_hash}")

        # Check existing indexation
        existing_count = self.check_existing_doc_hash(doc_hash)
        if existing_count > 0:
            if not force_reindex:
                logger.info(
                    f"Document with hash {doc_hash[:10]} is already indexed ({existing_count} records). Skipping."
                )
                return {
                    "stats": {
                        "document": self.pdf_path.name,
                        "doc_hash": doc_hash,
                        "status": "already_indexed",
                        "records_count": existing_count,
                        "table_name": self.table_name,
                    }
                }
            else:
                logger.info(f"Existing document detected with {existing_count} records. Purging for clean reindex...")
                self.delete_existing_doc(doc_hash)

        # 2. Compute Joint Multimodal CLIP Embeddings
        # Texts/Tables -> Text encoder
        # Images -> Vision encoder (directly from pixels!)
        logger.info(f"Generating joint multimodal CLIP embeddings for {len(nodes)} nodes...")
        for node in nodes:
            node_type = node.metadata.get("type")
            if node_type == "image":
                img_path = node.metadata.get("image_path")
                if img_path and Path(img_path).exists():
                    emb = self.clip_manager.embed_image(img_path)
                    node.embedding = emb
                else:
                    # Fallback to text embedding if image missing
                    node.embedding = self.clip_manager.embed_text(node.get_content())
            else:
                # Text chunk or table markdown
                node.embedding = self.clip_manager.embed_text(node.get_content())

        # 3. Initialize Storage Context with PGVector
        vector_store = get_vector_store(table_name=self.table_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = get_embedding_model()

        # 4. Insert into PGVector
        logger.info(f"Storing {len(nodes)} multimodal nodes into PGVector table '{self.table_name}'...")
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
        logger.info(f"Multimodal nodes successfully indexed into PGVector!")

        stats = {
            "document": self.pdf_path.name,
            "doc_hash": doc_hash,
            "total_nodes": len(nodes),
            "text_nodes": sum(1 for n in nodes if n.metadata.get("type") == "text"),
            "table_nodes": sum(1 for n in nodes if n.metadata.get("type") == "table"),
            "image_nodes": sum(1 for n in nodes if n.metadata.get("type") == "image"),
            "extracted_images_count": len(images_meta),
            "output_image_dir": str(self.output_image_dir.resolve()),
            "table_name": self.table_name,
            "embedding_dimension": self.clip_manager.dimension,
        }

        return {
            "index": index,
            "stats": stats,
            "images_meta": images_meta,
            "nodes": nodes,
        }
