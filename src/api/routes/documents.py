"""Document CRUD endpoints."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from ..schemas import (
    DocumentSummary,
    DocumentDetail,
    MarkdownResponse,
    UploadResponse,
)
from ..dependencies import get_sqlite_store, get_pipeline, get_config
from ...config import Config
from ...load import SQLiteStore
from ...pipeline import Pipeline
from ...models import DocumentStatus

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    status: Optional[str] = None,
    limit: int = 100,
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """
    List all documents.

    - **status**: Optional filter by status (pending, processing, completed, failed)
    - **limit**: Maximum number of documents to return
    """
    status_enum = DocumentStatus(status) if status else None
    documents = store.list_documents(status_enum)

    return [
        DocumentSummary(
            id=doc.id,
            filename=doc.filename,
            title=doc.title,
            status=doc.status.value,
            page_count=doc.page_count,
            uploaded_at=doc.uploaded_at,
            processed_at=doc.processed_at,
        )
        for doc in documents[:limit]
    ]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """Get document details by ID."""
    doc = store.get_document(document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    # Get chunk and image counts
    chunk_count = len(store.get_chunks_by_document(document_id)) if hasattr(store, 'get_chunks_by_document') else 0
    image_count = len(store.get_images_by_document(document_id)) if hasattr(store, 'get_images_by_document') else 0

    return DocumentDetail(
        id=doc.id,
        filename=doc.filename,
        title=doc.title,
        status=doc.status.value,
        page_count=doc.page_count,
        uploaded_at=doc.uploaded_at,
        processed_at=doc.processed_at,
        file_hash=doc.file_hash,
        source_path=doc.source_path,
        markdown_path=doc.markdown_path or "",
        extraction_method=doc.extraction_method or "",
        error_message=doc.error_message,
        chunk_count=chunk_count,
        image_count=image_count,
    )


@router.get("/{document_id}/markdown", response_model=MarkdownResponse)
def get_document_markdown(
    document_id: str,
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """Get extracted markdown content for a document."""
    doc = store.get_document(document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    if not doc.markdown_path:
        raise HTTPException(status_code=404, detail="Markdown file not available")

    markdown_path = Path(doc.markdown_path)
    if not markdown_path.exists():
        raise HTTPException(status_code=404, detail="Markdown file not found on disk")

    content = markdown_path.read_text(encoding="utf-8")

    return MarkdownResponse(
        document_id=doc.id,
        filename=doc.filename,
        content=content,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    force: bool = False,
    pipeline: Pipeline = Depends(get_pipeline),
    config: Config = Depends(get_config),
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """
    Upload and process a PDF document.

    This endpoint processes the document synchronously and returns when complete.
    Duplicate files (same content) are skipped unless force=true.

    - **file**: PDF file to upload
    - **force**: If true, reprocess even if file already exists
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save uploaded file
    pdf_path = config.paths.pdf_dir / file.filename
    config.paths.pdf_dir.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()

        # Check for duplicate by file hash
        import hashlib
        file_hash = hashlib.sha256(content).hexdigest()
        existing = store.get_document_by_hash(file_hash)

        if existing and not force:
            return UploadResponse(
                document_id=existing.id,
                status="skipped",
                message=f"Document already exists: {existing.filename} (use force=true to reprocess)",
            )

        # If force and exists, delete old document first
        if existing and force:
            pipeline.delete_document(existing.id)

        with open(pdf_path, "wb") as f:
            f.write(content)

        # Process document
        doc_id = pipeline.process_document(pdf_path)

        return UploadResponse(
            document_id=doc_id,
            status="completed",
            message=f"Document processed successfully: {file.filename}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    pipeline: Pipeline = Depends(get_pipeline),
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """Delete a document and all associated data."""
    doc = store.get_document(document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    try:
        pipeline.delete_document(document_id)
        return {"message": f"Document deleted: {document_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
