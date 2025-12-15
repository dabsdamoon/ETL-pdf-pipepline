# Storage Backend Switching Plan

## Overview

Implement an abstraction layer that allows switching between local storage (SQLite + LanceDB) and cloud storage (Supabase + pgvector) via configuration.

---

## Goals

1. **Zero code changes** when switching backends - configuration only
2. **Consistent API** across all storage backends
3. **Easy to extend** with additional backends (e.g., PostgreSQL, MongoDB)
4. **Graceful fallback** and clear error messages

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│                  (Pipeline, API, Retriever)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Abstraction                       │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ MetadataStore   │  │  VectorStore    │                   │
│  │   (Protocol)    │  │   (Protocol)    │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
└───────────┼────────────────────┼─────────────────────────────┘
            │                    │
     ┌──────┴──────┐      ┌──────┴──────┐
     ▼             ▼      ▼             ▼
┌─────────┐  ┌──────────┐ ┌─────────┐  ┌──────────┐
│ SQLite  │  │ Supabase │ │ LanceDB │  │ pgvector │
│ (local) │  │ (cloud)  │ │ (local) │  │ (cloud)  │
└─────────┘  └──────────┘ └─────────┘  └──────────┘
```

---

## Configuration

### Environment Variables

```bash
# Storage backend selection
STORAGE_BACKEND=local          # Options: local, supabase

# Local backend (default)
SQLITE_PATH=data/sqlite/etl.db
LANCEDB_PATH=data/lancedb

# Supabase backend (when STORAGE_BACKEND=supabase)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_DB_URL=postgresql://...  # For direct DB access (pgvector)
```

### Config Class Updates

```python
# src/config.py

class StorageConfig:
    backend: str = "local"  # local | supabase

    # Local settings
    sqlite_path: Path
    lancedb_path: Path

    # Cloud settings (Supabase)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_db_url: Optional[str] = None
```

---

## Implementation Plan

### Phase 1: Define Storage Protocols

Create abstract interfaces that both backends must implement.

**Files to create:**
- `src/storage/__init__.py`
- `src/storage/protocols.py`

```python
# src/storage/protocols.py

from typing import Protocol, Optional
from ..models import Document, DocumentStatus, Chunk, ExtractedImage

class MetadataStore(Protocol):
    """Protocol for document/chunk/image metadata storage."""

    # Document operations
    def insert_document(self, document: Document) -> None: ...
    def get_document(self, document_id: str) -> Optional[Document]: ...
    def get_document_by_hash(self, file_hash: str) -> Optional[Document]: ...
    def list_documents(self, status: Optional[DocumentStatus] = None) -> list[Document]: ...
    def update_document_status(self, document_id: str, status: DocumentStatus, error_message: Optional[str] = None) -> None: ...
    def delete_document(self, document_id: str) -> None: ...

    # Chunk operations
    def insert_chunks(self, chunks: list[Chunk]) -> None: ...
    def get_chunks_by_document(self, document_id: str) -> list[dict]: ...
    def delete_chunks(self, document_id: str) -> None: ...

    # Image operations
    def insert_images(self, images: list[ExtractedImage]) -> None: ...
    def get_images_by_document(self, document_id: str) -> list[ExtractedImage]: ...
    def get_image(self, image_id: str) -> Optional[ExtractedImage]: ...
    def delete_images(self, document_id: str) -> None: ...

    # Lifecycle
    def close(self) -> None: ...


class VectorStore(Protocol):
    """Protocol for vector storage and similarity search."""

    def insert_chunks(self, chunks: list[Chunk]) -> None: ...
    def get_chunk(self, chunk_id: str) -> Optional[dict]: ...
    def get_chunks_by_document(self, document_id: str, limit: int = 100) -> list[dict]: ...
    def delete_chunks(self, document_id: str) -> None: ...

    # Search
    def vector_search(self, query_embedding: list[float], limit: int = 10, title_filter: Optional[str] = None) -> list[dict]: ...
    def keyword_search(self, query: str, limit: int = 10, title_filter: Optional[str] = None) -> list[dict]: ...

    def close(self) -> None: ...
```

---

### Phase 2: Refactor Existing Stores

Ensure current SQLiteStore and LanceDBStore conform to the protocols.

**Files to modify:**
- `src/load/sqlite_store.py` - Ensure all Protocol methods exist
- `src/load/lancedb_store.py` - Ensure all Protocol methods exist

**Changes needed:**

1. **SQLiteStore**: Already implements most methods. Verify method signatures match Protocol.

2. **LanceDBStore**: Add missing methods if any:
   - Verify `keyword_search` exists
   - Verify `vector_search` signature matches

---

### Phase 3: Implement Supabase Backend

**Files to create:**
- `src/storage/supabase/__init__.py`
- `src/storage/supabase/metadata_store.py`
- `src/storage/supabase/vector_store.py`

#### Supabase Metadata Store

```python
# src/storage/supabase/metadata_store.py

from supabase import create_client, Client
from ...models import Document, DocumentStatus, Chunk, ExtractedImage

class SupabaseMetadataStore:
    """Supabase-based metadata storage."""

    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def insert_document(self, document: Document) -> None:
        self.client.table("documents").insert(document.to_dict()).execute()

    def get_document(self, document_id: str) -> Optional[Document]:
        result = self.client.table("documents").select("*").eq("id", document_id).execute()
        if result.data:
            return Document.from_dict(result.data[0])
        return None

    # ... implement remaining methods
```

#### Supabase Vector Store (pgvector)

```python
# src/storage/supabase/vector_store.py

import psycopg2
from pgvector.psycopg2 import register_vector

class SupabaseVectorStore:
    """pgvector-based vector storage via Supabase PostgreSQL."""

    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url)
        register_vector(self.conn)

    def vector_search(self, query_embedding: list[float], limit: int = 10, title_filter: Optional[str] = None) -> list[dict]:
        cursor = self.conn.cursor()

        query = """
            SELECT id, document_id, document_title, text, section_h1, section_h2,
                   page_numbers, chunk_index,
                   1 - (embedding <=> %s::vector) as score
            FROM chunks
            WHERE ($1 IS NULL OR document_title ILIKE $2)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cursor.execute(query, (query_embedding, title_filter, f"%{title_filter}%" if title_filter else None, query_embedding, limit))
        # ... return results
```

---

### Phase 4: Storage Factory

Create a factory to instantiate the correct backend based on configuration.

**Files to create:**
- `src/storage/factory.py`

```python
# src/storage/factory.py

from ..config import Config
from .protocols import MetadataStore, VectorStore

def create_metadata_store(config: Config) -> MetadataStore:
    """Factory to create the appropriate metadata store."""
    backend = config.storage.backend

    if backend == "local":
        from ..load import SQLiteStore
        return SQLiteStore(config)

    elif backend == "supabase":
        from .supabase import SupabaseMetadataStore
        return SupabaseMetadataStore(
            url=config.storage.supabase_url,
            key=config.storage.supabase_key,
        )

    else:
        raise ValueError(f"Unknown storage backend: {backend}")


def create_vector_store(config: Config) -> VectorStore:
    """Factory to create the appropriate vector store."""
    backend = config.storage.backend

    if backend == "local":
        from ..load import LanceDBStore
        return LanceDBStore(config)

    elif backend == "supabase":
        from .supabase import SupabaseVectorStore
        return SupabaseVectorStore(
            db_url=config.storage.supabase_db_url,
        )

    else:
        raise ValueError(f"Unknown storage backend: {backend}")
```

---

### Phase 5: Update Application Layer

Modify Pipeline, API, and Retriever to use the factory instead of direct imports.

**Files to modify:**
- `src/pipeline.py`
- `src/api/main.py`
- `src/api/dependencies.py`
- `src/retrieve/hybrid_retriever.py`

#### Pipeline Changes

```python
# src/pipeline.py

from .storage.factory import create_metadata_store, create_vector_store

class Pipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()

        # Use factory instead of direct instantiation
        self.metadata_store = create_metadata_store(self.config)
        self.vector_store = create_vector_store(self.config)
```

#### API Changes

```python
# src/api/main.py

from ..storage.factory import create_metadata_store, create_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()
    app.state.config = config

    # Use factory
    app.state.metadata_store = create_metadata_store(config)
    app.state.vector_store = create_vector_store(config)
    # ...
```

---

### Phase 6: Database Migrations (Supabase)

SQL migrations to create tables in Supabase.

**Files to create:**
- `migrations/supabase/001_create_tables.sql`
- `migrations/supabase/002_enable_pgvector.sql`

```sql
-- migrations/supabase/001_create_tables.sql

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    title TEXT,
    file_hash TEXT NOT NULL UNIQUE,
    page_count INTEGER,
    status TEXT DEFAULT 'pending',
    extraction_method TEXT,
    fallback_reason TEXT,
    source_path TEXT,
    markdown_path TEXT,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_hash ON documents(file_hash);

-- Images table
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    image_index INTEGER,
    file_path TEXT,
    width INTEGER,
    height INTEGER,
    format TEXT,
    caption TEXT,
    position JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_images_document ON images(document_id);
```

```sql
-- migrations/supabase/002_enable_pgvector.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table with vector column
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    document_title TEXT,
    text TEXT NOT NULL,
    section_h1 TEXT,
    section_h2 TEXT,
    section_h3 TEXT,
    page_numbers INTEGER[],
    chunk_index INTEGER,
    token_count INTEGER,
    embedding vector(384),  -- Dimension for all-MiniLM-L6-v2
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON chunks(document_id);

-- Create HNSW index for fast similarity search
CREATE INDEX idx_chunks_embedding ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index for keyword search
CREATE INDEX idx_chunks_text_search ON chunks
USING gin (to_tsvector('english', text));
```

---

### Phase 7: Image Storage (Cloud)

For cloud deployment, images need cloud storage (Supabase Storage or S3).

**Options:**
1. **Supabase Storage** - Built-in, easy integration
2. **AWS S3** - More flexible, requires additional setup
3. **Cloudflare R2** - S3-compatible, cost-effective

**Files to create:**
- `src/storage/supabase/image_storage.py`

```python
# src/storage/supabase/image_storage.py

class SupabaseImageStorage:
    """Store images in Supabase Storage bucket."""

    def __init__(self, client: Client, bucket_name: str = "images"):
        self.client = client
        self.bucket = bucket_name

    def upload(self, file_path: str, content: bytes) -> str:
        """Upload image and return public URL."""
        result = self.client.storage.from_(self.bucket).upload(file_path, content)
        return self.client.storage.from_(self.bucket).get_public_url(file_path)

    def download(self, file_path: str) -> bytes:
        """Download image content."""
        return self.client.storage.from_(self.bucket).download(file_path)

    def delete(self, file_path: str) -> None:
        """Delete image."""
        self.client.storage.from_(self.bucket).remove([file_path])
```

---

## File Structure

```
src/
├── storage/
│   ├── __init__.py
│   ├── protocols.py          # Abstract interfaces
│   ├── factory.py            # Backend factory
│   └── supabase/
│       ├── __init__.py
│       ├── metadata_store.py # Supabase metadata implementation
│       ├── vector_store.py   # pgvector implementation
│       └── image_storage.py  # Supabase Storage implementation
├── load/
│   ├── sqlite_store.py       # (existing, implements MetadataStore)
│   └── lancedb_store.py      # (existing, implements VectorStore)
migrations/
└── supabase/
    ├── 001_create_tables.sql
    └── 002_enable_pgvector.sql
```

---

## Dependencies

```txt
# requirements.txt additions (for Supabase backend)
supabase>=2.0.0
psycopg2-binary>=2.9.0
pgvector>=0.2.0
```

---

## Testing Strategy

1. **Unit Tests**: Test each store implementation independently
2. **Integration Tests**: Test factory creates correct backends
3. **E2E Tests**: Run full pipeline with both backends

```python
# tests/test_storage_factory.py

def test_local_backend():
    config = Config()
    config.storage.backend = "local"

    metadata = create_metadata_store(config)
    assert isinstance(metadata, SQLiteStore)

def test_supabase_backend():
    config = Config()
    config.storage.backend = "supabase"
    config.storage.supabase_url = "https://xxx.supabase.co"
    config.storage.supabase_key = "test-key"

    metadata = create_metadata_store(config)
    assert isinstance(metadata, SupabaseMetadataStore)
```

---

## Migration Path

### From Local to Cloud

1. Export data from local SQLite/LanceDB
2. Run Supabase migrations
3. Import data to Supabase
4. Update `.env` to set `STORAGE_BACKEND=supabase`
5. Restart application

**Migration script:**
```python
# scripts/migrate_to_supabase.py

def migrate():
    local_config = Config()
    local_config.storage.backend = "local"

    cloud_config = Config()
    cloud_config.storage.backend = "supabase"

    local_meta = create_metadata_store(local_config)
    cloud_meta = create_metadata_store(cloud_config)

    # Migrate documents
    for doc in local_meta.list_documents():
        cloud_meta.insert_document(doc)

    # Migrate chunks, images...
```

---

## Implementation Order

1. **Phase 1**: Define protocols (~1 file)
2. **Phase 2**: Refactor existing stores (~2 files, minor changes)
3. **Phase 3**: Implement Supabase backend (~3 files)
4. **Phase 4**: Create factory (~1 file)
5. **Phase 5**: Update application layer (~4 files)
6. **Phase 6**: Database migrations (~2 SQL files)
7. **Phase 7**: Image storage (~1 file)

---

## Considerations

### Hybrid Search on Supabase

pgvector supports vector search, but BM25 keyword search requires:
- **Option A**: Use PostgreSQL full-text search (`to_tsvector`, `ts_rank`)
- **Option B**: Use Supabase Edge Functions with external search
- **Option C**: Implement RRF (Reciprocal Rank Fusion) in application code

**Recommendation**: Use PostgreSQL full-text search for simplicity.

### Connection Pooling

For production Supabase usage:
- Use connection pooling (Supabase provides PgBouncer)
- Configure pool size based on expected load

### Error Handling

Both backends should raise consistent exceptions:
```python
class StorageError(Exception):
    """Base exception for storage errors."""
    pass

class DocumentNotFoundError(StorageError):
    pass

class DuplicateDocumentError(StorageError):
    pass
```
