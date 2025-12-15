"""Chunk endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import ChunkResponse
from ..dependencies import get_sqlite_store, get_lancedb_store
from ...load import SQLiteStore, LanceDBStore

router = APIRouter(tags=["chunks"])


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkResponse])
def get_document_chunks(
    document_id: str,
    limit: int = 100,
    store: SQLiteStore = Depends(get_sqlite_store),
    lancedb: LanceDBStore = Depends(get_lancedb_store),
):
    """Get all chunks for a document."""
    # Verify document exists
    doc = store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    # Get chunks from LanceDB (has full text)
    try:
        chunks = lancedb.get_chunks_by_document(document_id, limit=limit)
    except Exception:
        # Fallback: return empty if method doesn't exist or fails
        chunks = []

    return [
        ChunkResponse(
            id=chunk.get("id", ""),
            document_id=chunk.get("document_id", ""),
            document_title=chunk.get("document_title", ""),
            text=chunk.get("text", ""),
            section_h1=chunk.get("section_h1"),
            section_h2=chunk.get("section_h2"),
            page_numbers=chunk.get("page_numbers", []),
            chunk_index=chunk.get("chunk_index", 0),
            token_count=chunk.get("token_count", 0) if "token_count" in chunk else len(chunk.get("text", "").split()),
        )
        for chunk in chunks
    ]


@router.get("/chunks/{chunk_id}", response_model=ChunkResponse)
def get_chunk(
    chunk_id: str,
    lancedb: LanceDBStore = Depends(get_lancedb_store),
):
    """Get a single chunk by ID."""
    try:
        chunk = lancedb.get_chunk(chunk_id)
    except Exception:
        chunk = None

    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

    return ChunkResponse(
        id=chunk.get("id", ""),
        document_id=chunk.get("document_id", ""),
        document_title=chunk.get("document_title", ""),
        text=chunk.get("text", ""),
        section_h1=chunk.get("section_h1"),
        section_h2=chunk.get("section_h2"),
        page_numbers=chunk.get("page_numbers", []),
        chunk_index=chunk.get("chunk_index", 0),
        token_count=chunk.get("token_count", 0) if "token_count" in chunk else len(chunk.get("text", "").split()),
    )
