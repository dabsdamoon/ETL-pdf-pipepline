"""Hybrid text chunking using markdown structure and recursive splitting."""

from typing import Optional
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..models import Chunk
from ..config import ChunkingConfig
from ..logging_config import logger


class HybridChunker:
    """
    Hybrid chunking strategy:
    1. First split by markdown headers (preserves document structure)
    2. Then split large sections using recursive character splitting
    """

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

        # Level 1: Markdown structure-aware splitting
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.config.markdown_headers,
            strip_headers=False,  # Keep headers in content for context
        )

        # Level 2: Recursive splitting for large sections
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentences
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
            length_function=len,
        )

    def chunk(
        self,
        markdown_text: str,
        document_id: str,
        document_title: str,
    ) -> list[Chunk]:
        """
        Split markdown text into chunks with metadata.

        Args:
            markdown_text: The markdown content to chunk
            document_id: ID of the source document
            document_title: Title of the source document

        Returns:
            List of Chunk objects with metadata
        """
        chunks = []

        # Step 1: Split by markdown headers
        try:
            md_docs = self.md_splitter.split_text(markdown_text)
        except Exception as e:
            logger.warning(
                f"Markdown splitting failed, using recursive only: {e}",
                extra={"document_id": document_id},
            )
            # Fallback to recursive splitting only
            md_docs = [type("Doc", (), {"page_content": markdown_text, "metadata": {}})()]

        for md_doc in md_docs:
            section_text = md_doc.page_content
            section_metadata = md_doc.metadata if hasattr(md_doc, "metadata") else {}

            # Extract section headers
            section_h1 = section_metadata.get("h1")
            section_h2 = section_metadata.get("h2")
            section_h3 = section_metadata.get("h3")

            # Step 2: Further split large sections
            if len(section_text) > self.config.chunk_size * 1.5:
                sub_texts = self.recursive_splitter.split_text(section_text)
            else:
                sub_texts = [section_text] if section_text.strip() else []

            # Step 3: Create Chunk objects
            for i, text in enumerate(sub_texts):
                if not text.strip():
                    continue

                chunk = Chunk(
                    document_id=document_id,
                    document_title=document_title,
                    text=text.strip(),
                    section_h1=section_h1,
                    section_h2=section_h2,
                    section_h3=section_h3,
                    chunk_index=len(chunks),
                    is_section_start=(i == 0),
                    token_count=self._estimate_tokens(text),
                )
                chunks.append(chunk)

        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.info(
            f"Created {len(chunks)} chunks for document",
            extra={"document_id": document_id, "chunk_count": len(chunks)},
        )

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (words * 1.3 for English)."""
        return int(len(text.split()) * 1.3)

    def chunk_batch(
        self,
        documents: list[tuple[str, str, str]],  # (text, doc_id, title)
    ) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of (markdown_text, document_id, document_title) tuples

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for text, doc_id, title in documents:
            chunks = self.chunk(text, doc_id, title)
            all_chunks.extend(chunks)
        return all_chunks
