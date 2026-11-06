import logging
from pathlib import Path
from typing import List, Union, Optional, Any
from PIL import Image
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from pydantic import Field, PrivateAttr
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)

class MultimodalCLIPEmbeddings:
    """
    Multimodal Embedding model based on CLIP (ViT-B-32 via sentence-transformers/open_clip)
    to jointly project text and images into the same 512-dimensional normalized latent space.
    """
    _instance: Optional["MultimodalCLIPEmbeddings"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Multimodal CLIP model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = EMBEDDING_DIM
        logger.info(f"Multimodal CLIP model '{model_name}' ready (dimension: {self.dimension}).")

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "MultimodalCLIPEmbeddings":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized L2 embedding for text."""
        emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate normalized L2 embeddings for a list of texts."""
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

    def embed_image(self, image_input: Union[str, Path, Image.Image]) -> List[float]:
        """Generate normalized L2 embedding for an image."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        else:
            img = image_input.convert("RGB")

        emb = self.model.encode(img, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    def embed_images(
        self, image_inputs: List[Union[str, Path, Image.Image]], batch_size: int = 16
    ) -> List[List[float]]:
        """Generate normalized L2 embeddings for multiple images."""
        images = []
        for inp in image_inputs:
            if isinstance(inp, (str, Path)):
                images.append(Image.open(str(inp)).convert("RGB"))
            else:
                images.append(inp.convert("RGB"))

        embs = self.model.encode(
            images,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()


class LlamaIndexCLIPEmbedding(BaseEmbedding):
    """
    Adapter class allowing LlamaIndex to use the Multimodal CLIP model
    for standard text & node indexation.
    """
    model_name: str = Field(default=EMBEDDING_MODEL_NAME)
    _clip: MultimodalCLIPEmbeddings = PrivateAttr()

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self._clip = MultimodalCLIPEmbeddings.get_instance(model_name=model_name)

    @classmethod
    def class_name(cls) -> str:
        return "LlamaIndexCLIPEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._clip.embed_text(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._clip.embed_text(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._clip.embed_texts(texts)


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> LlamaIndexCLIPEmbedding:
    """Initialize and configure the Multimodal CLIP model for LlamaIndex."""
    clip_embed = LlamaIndexCLIPEmbedding(model_name=model_name)
    Settings.embed_model = clip_embed
    return clip_embed

def get_clip_manager() -> MultimodalCLIPEmbeddings:
    """Return the raw MultimodalCLIPEmbeddings instance for direct multimodal encoding."""
    return MultimodalCLIPEmbeddings.get_instance()
