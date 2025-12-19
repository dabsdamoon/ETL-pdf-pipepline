"""ETL Pipeline for PDF Document Processing.

A comprehensive ETL pipeline for extracting, transforming, and loading PDF documents
into vector databases for RAG (Retrieval-Augmented Generation) applications.

Basic Usage:
    >>> from etl_pdf_pipeline import Pipeline
    >>> pipeline = Pipeline()
    >>> doc_id = pipeline.process_document("path/to/document.pdf")

Functional API:
    >>> from etl_pdf_pipeline import extract_pdf, chunk_text, embed_chunks
    >>> markdown, metadata = extract_pdf("document.pdf")
    >>> chunks = chunk_text(markdown, document_id="doc1", title="My Document")
    >>> embedded_chunks = embed_chunks(chunks)

Components:
    - Extract: PDFValidator, PyMuPDFExtractor, ImageExtractor
    - Transform: MarkdownParser, HybridChunker, Embedder
    - Load: SQLiteStore, LanceDBStore
"""

__version__ = "0.1.0"

# Core pipeline
from .pipeline import Pipeline

# Models
from .models import (
    Document,
    DocumentStatus,
    Chunk,
    ExtractedImage,
    SearchResult,
    ValidationResult,
    ExtractionQualityMetrics,
)

# Configuration
from .config import Config, PathConfig, ExtractionConfig, ChunkingConfig, EmbeddingConfig

# Exceptions
from .exceptions import (
    PipelineError,
    ValidationError,
    ExtractionError,
    TransformationError,
    LoadError,
)

# Extract components
from .extract import PDFValidator, PyMuPDFExtractor, ImageExtractor, GoogleVisionExtractor, GOOGLE_VISION_AVAILABLE

# Transform components
from .transform import MarkdownParser, HybridChunker, Embedder

# Load components
from .load import SQLiteStore, LanceDBStore

# Logging
from .logging_config import logger, setup_logging


# =============================================================================
# Convenience Functions (Functional API)
# =============================================================================

def extract_pdf(
    pdf_path: str,
    document_id: str = None,
    config: "Config" = None,
) -> tuple[str, dict]:
    """
    Extract text from a PDF file to markdown.

    Args:
        pdf_path: Path to the PDF file
        document_id: Optional document identifier (generated if not provided)
        config: Optional configuration object
            - Set config.extraction.method = "google_vision" for OCR extraction
            - Requires GOOGLE_APPLICATION_CREDENTIALS env variable

    Returns:
        Tuple of (markdown_content, metadata_dict)

    Example:
        >>> markdown, metadata = extract_pdf("document.pdf")
        >>> print(metadata["title"])
        >>> print(markdown[:500])

        # With Google Vision OCR:
        >>> from etl_pdf_pipeline import Config, ExtractionConfig
        >>> config = Config(extraction=ExtractionConfig(method="google_vision"))
        >>> markdown, metadata = extract_pdf("scanned.pdf", config=config)
    """
    import uuid
    from pathlib import Path

    cfg = config if config is not None else Config()

    # Select extractor based on config
    if cfg.extraction.method == "google_vision":
        if not GOOGLE_VISION_AVAILABLE:
            raise ImportError(
                "google-cloud-vision is required for Google Vision OCR. "
                "Install with: pip install google-cloud-vision"
            )
        extractor = GoogleVisionExtractor(cfg)
    else:
        extractor = PyMuPDFExtractor(cfg)

    pdf_path = Path(pdf_path)
    doc_id = document_id if document_id is not None else str(uuid.uuid4())

    document, markdown_path = extractor.extract_to_markdown(pdf_path, doc_id)

    # Read the markdown content
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()

    metadata = {
        "document_id": document.id,
        "title": document.title,
        "filename": document.filename,
        "page_count": document.page_count,
        "file_hash": document.file_hash,
        "markdown_path": str(markdown_path),
        "extraction_method": cfg.extraction.method,
    }

    return content, metadata


def chunk_text(
    text: str,
    document_id: str = None,
    title: str = "",
    config: Config = None,
) -> list[Chunk]:
    """
    Chunk text content using hybrid markdown-aware chunking.

    Args:
        text: Text or markdown content to chunk
        document_id: Optional document identifier
        title: Optional document title
        config: Optional configuration object

    Returns:
        List of Chunk objects

    Example:
        >>> chunks = chunk_text(markdown_content, document_id="doc1", title="My Doc")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {len(chunk.text)} chars")
    """
    import uuid

    config = config or Config()
    chunker = HybridChunker(config.chunking)

    document_id = document_id or str(uuid.uuid4())

    return chunker.chunk(text, document_id, title)


def embed_chunks(
    chunks: list[Chunk],
    config: Config = None,
) -> list[Chunk]:
    """
    Generate embeddings for chunks.

    Args:
        chunks: List of Chunk objects
        config: Optional configuration object

    Returns:
        List of Chunk objects with embeddings populated

    Example:
        >>> embedded = embed_chunks(chunks)
        >>> print(f"Embedding dimension: {len(embedded[0].embedding)}")
    """
    config = config or Config()
    embedder = Embedder(config.embedding)

    return embedder.embed_chunks(chunks)


def process_pdf(
    pdf_path: str,
    config: Config = None,
) -> tuple[list[Chunk], dict]:
    """
    Full ETL pipeline: extract, chunk, and embed a PDF.

    This is a convenience function that runs the complete pipeline
    without loading to any database, returning the embedded chunks
    for custom storage.

    Args:
        pdf_path: Path to the PDF file
        config: Optional configuration object

    Returns:
        Tuple of (list of embedded Chunk objects, metadata dict)

    Example:
        >>> chunks, metadata = process_pdf("document.pdf")
        >>> print(f"Processed {len(chunks)} chunks from {metadata['title']}")
        >>> # Now load to your own database
        >>> for chunk in chunks:
        ...     my_db.insert(chunk.to_dict())
    """
    config = config or Config()

    # Extract
    markdown, metadata = extract_pdf(pdf_path, config)

    # Chunk
    chunks = chunk_text(
        markdown,
        document_id=metadata["document_id"],
        title=metadata["title"],
        config=config,
    )

    # Embed
    embedded_chunks = embed_chunks(chunks, config)

    return embedded_chunks, metadata


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core
    "Pipeline",
    # Models
    "Document",
    "DocumentStatus",
    "Chunk",
    "ExtractedImage",
    "SearchResult",
    "ValidationResult",
    "ExtractionQualityMetrics",
    # Config
    "Config",
    "PathConfig",
    "ExtractionConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    # Exceptions
    "PipelineError",
    "ValidationError",
    "ExtractionError",
    "TransformationError",
    "LoadError",
    # Extract
    "PDFValidator",
    "PyMuPDFExtractor",
    "ImageExtractor",
    "GoogleVisionExtractor",
    "GOOGLE_VISION_AVAILABLE",
    # Transform
    "MarkdownParser",
    "HybridChunker",
    "Embedder",
    # Load
    "SQLiteStore",
    "LanceDBStore",
    # Logging
    "logger",
    "setup_logging",
    # Convenience functions
    "extract_pdf",
    "chunk_text",
    "embed_chunks",
    "process_pdf",
]
