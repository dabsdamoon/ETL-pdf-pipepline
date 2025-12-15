"""Image endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..schemas import ImageResponse
from ..dependencies import get_sqlite_store
from ...load import SQLiteStore

router = APIRouter(tags=["images"])


@router.get("/documents/{document_id}/images", response_model=list[ImageResponse])
def get_document_images(
    document_id: str,
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """Get all image metadata for a document."""
    # Verify document exists
    doc = store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    try:
        images = store.get_images_by_document(document_id)
    except Exception:
        images = []

    return [
        ImageResponse(
            id=img.id,
            document_id=img.document_id,
            page_number=img.page_number,
            image_index=img.image_index,
            file_path=img.file_path,
            width=img.width,
            height=img.height,
            format=img.format,
            caption=img.caption,
        )
        for img in images
    ]


@router.get("/images/{image_id}", response_model=ImageResponse)
def get_image(
    image_id: str,
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """Get image metadata by ID."""
    try:
        img = store.get_image(image_id)
    except Exception:
        img = None

    if img is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    return ImageResponse(
        id=img.id,
        document_id=img.document_id,
        page_number=img.page_number,
        image_index=img.image_index,
        file_path=img.file_path,
        width=img.width,
        height=img.height,
        format=img.format,
        caption=img.caption,
    )


@router.get("/images/{image_id}/file")
def get_image_file(
    image_id: str,
    store: SQLiteStore = Depends(get_sqlite_store),
):
    """Serve the actual image file."""
    try:
        img = store.get_image(image_id)
    except Exception:
        img = None

    if img is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    file_path = Path(img.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Determine media type
    media_type_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    media_type = media_type_map.get(img.format.lower(), "application/octet-stream")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )
