"""Embedding generation for text chunks."""

from typing import Optional
import os

from ..models import Chunk
from ..config import EmbeddingConfig
from ..logging_config import logger
from ..exceptions import TransformationError


class Embedder:
    """
    Generates embeddings for text chunks.

    Supports:
    - Local: sentence-transformers (all-MiniLM-L6-v2)
    - API: OpenAI (text-embedding-3-small)
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._openai_client = None

    def _get_local_model(self):
        """Lazy load the local sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.config.local_model}")
            self._model = SentenceTransformer(self.config.local_model)
        return self._model

    def _get_openai_client(self):
        """Lazy load the OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise TransformationError(
                    "OPENAI_API_KEY not set but OpenAI embeddings requested",
                    phase="embedding",
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.config.provider == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local sentence-transformers model."""
        model = self._get_local_model()

        logger.info(f"Generating {len(texts)} embeddings with local model")
        embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)

        return [emb.tolist() for emb in embeddings]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        client = self._get_openai_client()

        all_embeddings = []
        batch_size = self.config.batch_size

        # Process in batches to respect rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"Generating embeddings batch {i // batch_size + 1} "
                f"({len(batch)} texts) with OpenAI"
            )

            try:
                response = client.embeddings.create(
                    model=self.config.openai_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                raise TransformationError(
                    f"OpenAI embedding API error: {e}",
                    phase="embedding",
                ) from e

        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Same chunks with embedding field populated
        """
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        logger.info(
            f"Generated embeddings for {len(chunks)} chunks",
            extra={"chunk_count": len(chunks)},
        )

        return chunks

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.config.dimension
