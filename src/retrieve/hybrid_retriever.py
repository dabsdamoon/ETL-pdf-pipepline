"""Hybrid retriever combining vector and BM25 search."""

from enum import Enum
from typing import Optional

from ..models import SearchResult
from ..config import Config
from ..load.lancedb_store import LanceDBStore
from ..transform.embedder import Embedder
from ..logging_config import logger


class SearchMode(Enum):
    """Search mode for retrieval."""

    VECTOR = "vector"  # Pure semantic search
    HYBRID = "hybrid"  # Vector + BM25 (recommended)
    KEYWORD = "keyword"  # Pure BM25


class HybridRetriever:
    """
    RAG retriever with hybrid search capabilities.

    Combines:
    - Vector search (semantic similarity)
    - BM25 search (keyword matching)
    - Document title routing
    """

    # Known topics for title-based routing
    KNOWN_TOPICS = [
        "pregnancy",
        "nutrition",
        "diabetes",
        "vaccination",
        "contraception",
        "menopause",
        "fertility",
        "labor",
        "cesarean",
        "breastfeeding",
        "depression",
        "exercise",
        "cancer",
        "incontinence",
        "hysterectomy",
        "infection",
        "bleeding",
    ]

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.store = LanceDBStore(config)
        self.embedder = Embedder(self.config.embedding)

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10,
        title_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for relevant chunks.

        Args:
            query: User's question or search query
            mode: Search mode (vector, hybrid, keyword)
            limit: Maximum number of results
            title_filter: Optional filter for document title

        Returns:
            List of SearchResult objects
        """
        # Build filter expression (with SQL injection protection)
        filter_expr = None
        if title_filter:
            # Escape single quotes to prevent SQL injection
            safe_filter = title_filter.replace("'", "''").replace("%", "\\%")
            filter_expr = f"document_title LIKE '%{safe_filter}%'"

        if mode == SearchMode.VECTOR:
            results = self._vector_search(query, limit, filter_expr)
        elif mode == SearchMode.KEYWORD:
            results = self._keyword_search(query, limit, filter_expr)
        else:  # HYBRID
            results = self._hybrid_search(query, limit, filter_expr)

        return [self._to_search_result(r, mode) for r in results]

    def search_with_routing(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Smart search with automatic title-based routing.

        Detects topic keywords in the query and uses them to filter results.

        Args:
            query: User's question
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        # Extract topic keywords from query
        topics = self._extract_topics(query)

        if topics:
            # Use first matching topic as filter
            logger.info(
                f"Routing to topic: {topics[0]}",
                extra={"topics": topics, "query": query[:50]},
            )
            return self.search(
                query=query,
                mode=SearchMode.HYBRID,
                limit=limit,
                title_filter=topics[0],
            )
        else:
            # No specific topic - full hybrid search
            return self.search(query=query, mode=SearchMode.HYBRID, limit=limit)

    def _vector_search(
        self, query: str, limit: int, filter_expr: Optional[str]
    ) -> list[dict]:
        """Pure vector search."""
        query_embedding = self.embedder.embed_texts([query])[0]
        return self.store.vector_search(query_embedding, limit, filter_expr)

    def _keyword_search(
        self, query: str, limit: int, filter_expr: Optional[str]
    ) -> list[dict]:
        """Pure BM25 keyword search."""
        return self.store.fts_search(query, limit, filter_expr)

    def _hybrid_search(
        self, query: str, limit: int, filter_expr: Optional[str]
    ) -> list[dict]:
        """Hybrid vector + BM25 search."""
        query_embedding = self.embedder.embed_texts([query])[0]
        return self.store.hybrid_search(query, query_embedding, limit, filter_expr)

    def _extract_topics(self, query: str) -> list[str]:
        """Extract known topic keywords from query."""
        query_lower = query.lower()
        return [topic for topic in self.KNOWN_TOPICS if topic in query_lower]

    def _to_search_result(self, record: dict, mode: SearchMode) -> SearchResult:
        """Convert a database record to SearchResult."""
        return SearchResult(
            chunk_id=record.get("id", ""),
            document_id=record.get("document_id", ""),
            document_title=record.get("document_title", ""),
            text=record.get("text", ""),
            page_numbers=record.get("page_numbers", []),
            score=record.get("_score", record.get("_distance", 0.0)),
            search_mode=mode.value,
            section_h1=record.get("section_h1"),
            section_h2=record.get("section_h2"),
        )

    def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> str:
        """
        Get formatted context for LLM prompt.

        Args:
            query: User's question
            max_tokens: Maximum tokens in context
            mode: Search mode

        Returns:
            Formatted context string with source attribution
        """
        results = self.search(query, mode=mode, limit=20)

        context_parts = []
        total_tokens = 0
        seen_docs = set()

        for r in results:
            # Rough token estimate
            tokens = len(r.text.split()) * 1.3
            if total_tokens + tokens > max_tokens:
                break

            seen_docs.add(r.document_title)
            context_parts.append(
                f"[Source: {r.document_title}]\n{r.text}"
            )
            total_tokens += tokens

        context = "\n\n---\n\n".join(context_parts)

        # Add document list
        doc_list = "\n".join(f"- {doc}" for doc in seen_docs)
        return f"Documents referenced:\n{doc_list}\n\n---\n\n{context}"
