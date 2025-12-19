"""Configuration management for the ETL pipeline."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PathConfig:
    """Path configuration for the pipeline."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    def __post_init__(self):
        self.pdf_dir = self.base_dir / "pdfs"
        self.markdown_dir = self.base_dir / "data" / "markdown"
        self.images_dir = self.base_dir / "data" / "images"
        self.lancedb_dir = self.base_dir / "data" / "lancedb"
        self.sqlite_dir = self.base_dir / "data" / "sqlite"
        self.logs_dir = self.base_dir / "logs"

        # Ensure directories exist
        for path in [
            self.markdown_dir,
            self.images_dir,
            self.lancedb_dir,
            self.sqlite_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def sqlite_path(self) -> Path:
        return self.sqlite_dir / "documents.db"


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction."""

    # Extraction method: "pymupdf" (default) or "google_vision"
    method: str = "pymupdf"

    # Google Vision OCR settings
    ocr_dpi: int = 300  # Resolution for page rendering

    # Quality thresholds for text density heuristics
    min_chars_per_page: int = 100
    min_words_per_page: int = 20
    max_non_ascii_ratio: float = 0.15
    min_avg_word_length: float = 2.0
    max_avg_word_length: float = 15.0

    # File limits
    max_file_size_mb: int = 50
    max_page_count: int = 500


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    use_semantic_chunking: bool = False  # Requires API calls
    markdown_headers: list = field(
        default_factory=lambda: [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: str = "local"  # "openai" or "local"
    openai_model: str = "text-embedding-3-small"
    local_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 100

    @property
    def dimension(self) -> int:
        if self.provider == "openai":
            return 1536
        return 384  # all-MiniLM-L6-v2


@dataclass
class Config:
    """Main configuration class."""

    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "local"))
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    paths: PathConfig = field(default_factory=PathConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def __post_init__(self):
        # Use OpenAI if API key is available and environment is production
        if self.openai_api_key and self.environment == "production":
            self.embedding.provider = "openai"


# Global config instance
config = Config()
