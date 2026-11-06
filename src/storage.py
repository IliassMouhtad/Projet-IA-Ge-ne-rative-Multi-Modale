import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from llama_index.vector_stores.postgres import PGVectorStore
from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    EMBEDDING_DIM,
    PG_TABLE_NAME,
)

logger = logging.getLogger(__name__)

def ensure_database_exists():
    """Ensure that the target database exists in PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()
        if not exists:
            logger.info(f"Database '{DB_NAME}' not found. Creating database...")
            cur.execute(f'CREATE DATABASE "{DB_NAME}"')
            logger.info(f"Database '{DB_NAME}' created.")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Database existence check/creation warning: {e}")

def get_vector_store(table_name: str = PG_TABLE_NAME) -> PGVectorStore:
    """Initialize and return the PGVectorStore."""
    ensure_database_exists()
    
    logger.info(f"Connecting to PGVectorStore (table: {table_name}, dim: {EMBEDDING_DIM})...")
    vector_store = PGVectorStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=table_name,
        embed_dim=EMBEDDING_DIM,
    )
    logger.info("PGVectorStore initialized successfully.")
    return vector_store
