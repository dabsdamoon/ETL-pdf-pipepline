"""API route modules."""

from .documents import router as documents_router
from .chunks import router as chunks_router
from .images import router as images_router
from .search import router as search_router
from .stats import router as stats_router

__all__ = [
    "documents_router",
    "chunks_router",
    "images_router",
    "search_router",
    "stats_router",
]
