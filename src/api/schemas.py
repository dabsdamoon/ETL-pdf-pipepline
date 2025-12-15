"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentSummary(BaseModel):
    """Document list item response."""

    id: str
    filename: str
    title: str
    status: str
    page_count: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentDetail(DocumentSummary):
    """Detailed document response."""

    file_hash: str
    source_path: str
    markdown_path: str
    extraction_method: str
    error_message: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0


class MarkdownResponse(BaseModel):
    """Markdown content response."""

    document_id: str
    filename: str
    content: str


class UploadResponse(BaseModel):
    """Document upload response."""

    document_id: str
    status: str
    message: str


# ============================================================================
# Chunk Schemas
# ============================================================================

class ChunkResponse(BaseModel):
    """Chunk response."""

    id: str
    document_id: str
    document_title: str
    text: str
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
    page_numbers: list[int]
    chunk_index: int
    token_count: int

    class Config:
        from_attributes = True


# ============================================================================
# Image Schemas
# ============================================================================

class ImageResponse(BaseModel):
    """Image metadata response."""

    id: str
    document_id: str
    page_number: int
    image_index: int
    file_path: str
    width: int
    height: int
    format: str
    caption: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================================
# Search Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """Search request."""

    query: str = Field(..., min_length=1, description="Search query")
    mode: str = Field(default="hybrid", description="Search mode: vector, hybrid, keyword")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    title_filter: Optional[str] = Field(default=None, description="Filter by document title")


class SearchResultResponse(BaseModel):
    """Search result item."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_numbers: list[int]
    score: float
    search_mode: str
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None


class ContextRequest(BaseModel):
    """Context generation request."""

    query: str = Field(..., min_length=1, description="Query for context")
    max_tokens: int = Field(default=4000, ge=100, le=16000, description="Max tokens in context")
    mode: str = Field(default="hybrid", description="Search mode")


class ContextResponse(BaseModel):
    """Context generation response."""

    context: str
    documents_referenced: list[str]


# ============================================================================
# Stats Schema
# ============================================================================

class StatsResponse(BaseModel):
    """Pipeline statistics response."""

    total_documents: int
    total_chunks: int
    by_status: dict[str, int]


# ============================================================================
# Error Schema
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_type: Optional[str] = None
