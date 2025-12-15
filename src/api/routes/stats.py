"""Statistics endpoint."""

from fastapi import APIRouter, Depends

from ..schemas import StatsResponse
from ..dependencies import get_pipeline
from ...pipeline import Pipeline

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Get pipeline statistics."""
    stats = pipeline.get_stats()

    return StatsResponse(
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
        by_status=stats["by_status"],
    )
