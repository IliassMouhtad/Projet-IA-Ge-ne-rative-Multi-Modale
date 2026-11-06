import os
import sys
import logging
from pprint import pprint
import psycopg2
from src.config import (
    DEFAULT_PDF_PATH,
    OUTPUT_IMAGE_DIR,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    PG_TABLE_NAME,
)
from src.indexer import MultimodalIndexer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_indexing")

def verify_pgvector_records():
    """Directly query PostgreSQL to verify stored vectors and metadata."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        cur = conn.cursor()
        
        # Check table existence (LlamaIndex creates table as data_{table_name})
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cur.fetchall()]
        logger.info(f"Existing PostgreSQL tables in '{DB_NAME}': {tables}")

        target_table = None
        for candidate in [f"data_{PG_TABLE_NAME}", PG_TABLE_NAME]:
            if candidate in tables:
                target_table = candidate
                break

        if target_table:
            cur.execute(f"SELECT count(*) FROM {target_table};")
            count = cur.fetchone()[0]
            logger.info(f"PGVector verification: Table '{target_table}' contains {count} vector records.")

            # Sample records by type
            cur.execute(f"""
                SELECT metadata_->>'type' as node_type, count(*) 
                FROM {target_table} 
                GROUP BY metadata_->>'type';
            """)
            type_distribution = cur.fetchall()
            logger.info(f"Node distribution by type in PGVector: {dict(type_distribution)}")
        else:
            logger.warning(f"Could not find table matching '{PG_TABLE_NAME}' in PostgreSQL.")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error querying PGVector for verification: {e}")

def main():
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF document not found at: {DEFAULT_PDF_PATH}")
        sys.exit(1)

    logger.info(f"Using input PDF: {DEFAULT_PDF_PATH}")
    indexer = MultimodalIndexer(
        pdf_path=DEFAULT_PDF_PATH,
        output_image_dir=OUTPUT_IMAGE_DIR,
        table_name=PG_TABLE_NAME,
    )

    result = indexer.run()
    stats = result["stats"]

    print("\n" + "=" * 60)
    print("      INDEXATION MULTIMODALE LLAMAINDEX + PGVECTOR")
    print("=" * 60)
    print(f"Document analysé           : {stats['document']}")
    print(f"Total nœuds indexés        : {stats['total_nodes']}")
    print(f" - Chunks de texte         : {stats['text_nodes']}")
    print(f" - Tableaux extraits       : {stats['table_nodes']}")
    print(f" - Références d'images     : {stats['image_nodes']}")
    print(f"Images extraites sur disque: {stats['extracted_images_count']}")
    print(f"Dossier des images         : {stats['output_image_dir']}")
    print(f"Table PGVector             : {stats['table_name']}")
    print("=" * 60 + "\n")

    logger.info("Verifying PostgreSQL PGVector database storage...")
    verify_pgvector_records()
    logger.info("Multimodal extraction and PGVector indexation finished successfully.")

if __name__ == "__main__":
    main()
