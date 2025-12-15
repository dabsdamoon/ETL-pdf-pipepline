"""Search endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    SearchRequest,
    SearchResultResponse,
    ContextRequest,
    ContextResponse,
)
from ..dependencies import get_retriever
from ...retrieve import HybridRetriever, SearchMode

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=list[SearchResultResponse])
def search(
    request: SearchRequest,
    retriever: HybridRetriever = Depends(get_retriever),
):
    """
    Search chunks using hybrid, vector, or keyword modes.

    - **query**: Search query text
    - **mode**: Search mode (vector, hybrid, keyword). Default: hybrid
    - **limit**: Maximum number of results. Default: 10
    - **title_filter**: Optional filter by document title
    """
    # Validate mode
    try:
        mode = SearchMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search mode: {request.mode}. Use: vector, hybrid, keyword",
        )

    results = retriever.search(
        query=request.query,
        mode=mode,
        limit=request.limit,
        title_filter=request.title_filter,
    )

    return [
        SearchResultResponse(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            document_title=r.document_title,
            text=r.text,
            page_numbers=r.page_numbers,
            score=r.score,
            search_mode=r.search_mode,
            section_h1=r.section_h1,
            section_h2=r.section_h2,
        )
        for r in results
    ]


@router.post("/context", response_model=ContextResponse)
def get_context(
    request: ContextRequest,
    retriever: HybridRetriever = Depends(get_retriever),
):
    """
    Get formatted context for LLM prompt.

    Returns chunked text with source attribution, respecting token budget.

    - **query**: Query for context retrieval
    - **max_tokens**: Maximum tokens in context. Default: 4000
    - **mode**: Search mode. Default: hybrid
    """
    try:
        mode = SearchMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search mode: {request.mode}",
        )

    context = retriever.get_context(
        query=request.query,
        max_tokens=request.max_tokens,
        mode=mode,
    )

    # Extract document references from context
    # Context format: "Documents referenced:\n- Doc1\n- Doc2\n\n---\n\n..."
    docs_referenced = []
    if context.startswith("Documents referenced:"):
        lines = context.split("\n\n---\n\n")[0].split("\n")
        docs_referenced = [
            line.strip("- ").strip()
            for line in lines[1:]
            if line.startswith("- ")
        ]

    return ContextResponse(
        context=context,
        documents_referenced=docs_referenced,
    )
