"""Data models for the ETL pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class DocumentStatus(Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTDATED = "outdated"


class ValidationResult(Enum):
    """Result of PDF validation."""

    VALID = "valid"
    FILE_TOO_LARGE = "file_too_large"
    TOO_MANY_PAGES = "too_many_pages"
    CORRUPTED = "corrupted"
    PASSWORD_PROTECTED = "password_protected"
    EMPTY = "empty"
    FILE_NOT_FOUND = "file_not_found"


@dataclass
class ExtractionQualityMetrics:
    """Metrics for evaluating extraction quality."""

    chars_per_page: float
    words_per_page: float
    avg_word_length: float
    whitespace_ratio: float
    non_ascii_ratio: float
    empty_pages_ratio: float = 0.0


@dataclass
class Document:
    """Represents a processed PDF document."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    title: str = ""
    file_hash: str = ""
    page_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    extraction_method: str = "pymupdf"
    fallback_reason: Optional[str] = None
    source_path: str = ""
    markdown_path: str = ""
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "title": self.title,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "status": self.status.value,
            "extraction_method": self.extraction_method,
            "fallback_reason": self.fallback_reason,
            "source_path": self.source_path,
            "markdown_path": self.markdown_path,
            "uploaded_at": self.uploaded_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "error_message": self.error_message,
        }


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_title: str = ""
    text: str = ""
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None
    page_numbers: list[int] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    is_section_start: bool = False
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "text": self.text,
            "section_h1": self.section_h1,
            "section_h2": self.section_h2,
            "section_h3": self.section_h3,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "is_section_start": self.is_section_start,
        }

    def to_lancedb_dict(self) -> dict:
        """Convert to LanceDB-compatible dict with embedding."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "text": self.text,
            "section_h1": self.section_h1 or "",
            "section_h2": self.section_h2 or "",
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "vector": self.embedding,
        }


@dataclass
class ExtractedImage:
    """Represents an extracted image from a document."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    page_number: int = 0
    image_index: int = 0
    file_path: str = ""
    width: int = 0
    height: int = 0
    format: str = "png"
    caption: Optional[str] = None
    position: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "image_index": self.image_index,
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "caption": self.caption,
            "position": self.position,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchResult:
    """Represents a search result from the retriever."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_numbers: list[int]
    score: float
    search_mode: str
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
