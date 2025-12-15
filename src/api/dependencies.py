"""FastAPI dependency injection for shared resources."""

from fastapi import Request

from ..config import Config
from ..load import SQLiteStore, LanceDBStore
from ..retrieve import HybridRetriever
from ..pipeline import Pipeline


def get_config(request: Request) -> Config:
    """Get configuration from app state."""
    return request.app.state.config


def get_sqlite_store(request: Request) -> SQLiteStore:
    """Get SQLite store from app state."""
    return request.app.state.sqlite_store


def get_lancedb_store(request: Request) -> LanceDBStore:
    """Get LanceDB store from app state."""
    return request.app.state.lancedb_store


def get_retriever(request: Request) -> HybridRetriever:
    """Get hybrid retriever from app state."""
    return request.app.state.retriever


def get_pipeline(request: Request) -> Pipeline:
    """Get pipeline from app state."""
    return request.app.state.pipeline
