"""FastAPI application for ETL Pipeline."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import Config
from ..pipeline import Pipeline
from ..load import SQLiteStore, LanceDBStore
from ..retrieve import HybridRetriever
from ..logging_config import setup_logging

from .routes import (
    documents_router,
    chunks_router,
    images_router,
    search_router,
    stats_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup: Initialize stores and resources
    config = Config()
    setup_logging(config.paths.logs_dir)

    app.state.config = config
    app.state.pipeline = Pipeline(config)
    app.state.sqlite_store = SQLiteStore(config)
    app.state.lancedb_store = LanceDBStore(config)
    app.state.retriever = HybridRetriever(config)

    yield

    # Shutdown: Close connections
    app.state.pipeline.close()
    app.state.sqlite_store.close()
    app.state.lancedb_store.close()


app = FastAPI(
    title="ETL Pipeline API",
    description="REST API for PDF document processing pipeline with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix="/api")
app.include_router(chunks_router, prefix="/api")
app.include_router(images_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(stats_router, prefix="/api")


@app.get("/")
def root():
    """Root endpoint - API information."""
    return {
        "name": "ETL Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
