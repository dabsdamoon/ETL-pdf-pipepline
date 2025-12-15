# ETL Pipeline API Reference

REST API for PDF document processing pipeline with RAG capabilities.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently no authentication is required (demo mode).

---

## Quick Start

```bash
# Start the API server
python scripts/run_api.py

# View interactive docs
open http://localhost:8000/docs
```

---

## Endpoints Overview

| Category | Method | Endpoint | Description |
|----------|--------|----------|-------------|
| Documents | GET | `/api/documents` | List all documents |
| Documents | GET | `/api/documents/{id}` | Get document details |
| Documents | GET | `/api/documents/{id}/markdown` | Get extracted markdown |
| Documents | POST | `/api/documents/upload` | Upload and process PDF |
| Documents | DELETE | `/api/documents/{id}` | Delete document |
| Chunks | GET | `/api/documents/{id}/chunks` | Get document chunks |
| Chunks | GET | `/api/chunks/{id}` | Get single chunk |
| Images | GET | `/api/documents/{id}/images` | Get document images |
| Images | GET | `/api/images/{id}` | Get image metadata |
| Images | GET | `/api/images/{id}/file` | Download image file |
| Search | POST | `/api/search` | Search chunks |
| Search | POST | `/api/search/context` | Get LLM context |
| Stats | GET | `/api/stats` | Get pipeline statistics |

---

## Documents

### List Documents

Retrieve a list of all documents with optional status filtering.

```
GET /api/documents
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by status: `pending`, `processing`, `completed`, `failed` |
| `limit` | integer | No | Maximum results (default: 100) |

**Response:** `200 OK`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "nutrition-guide.pdf",
    "title": "Nutrition During Pregnancy",
    "status": "completed",
    "page_count": 12,
    "uploaded_at": "2025-01-15T10:30:00Z",
    "processed_at": "2025-01-15T10:30:45Z"
  }
]
```

**Example:**

```bash
# List all documents
curl http://localhost:8000/api/documents

# List only completed documents
curl "http://localhost:8000/api/documents?status=completed"
```

---

### Get Document

Retrieve detailed information about a specific document.

```
GET /api/documents/{id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Document UUID |

**Response:** `200 OK`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "nutrition-guide.pdf",
  "title": "Nutrition During Pregnancy",
  "status": "completed",
  "page_count": 12,
  "uploaded_at": "2025-01-15T10:30:00Z",
  "processed_at": "2025-01-15T10:30:45Z",
  "file_hash": "sha256:abc123...",
  "source_path": "/data/pdfs/nutrition-guide.pdf",
  "markdown_path": "/data/markdown/550e8400-e29b-41d4-a716-446655440000.md",
  "extraction_method": "pymupdf",
  "error_message": null,
  "chunk_count": 45,
  "image_count": 8
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| `404` | Document not found |

---

### Get Document Markdown

Retrieve the extracted markdown content for a document.

```
GET /api/documents/{id}/markdown
```

**Response:** `200 OK`

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "nutrition-guide.pdf",
  "content": "---\ndocument_id: \"550e8400...\"\ntitle: \"Nutrition During Pregnancy\"\n---\n\n# Nutrition During Pregnancy\n\n## Why Nutrition Matters\n\nGood nutrition during pregnancy..."
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| `404` | Document or markdown file not found |

---

### Upload Document

Upload and process a new PDF document. Processing is synchronous.

```
POST /api/documents/upload
```

**Request:**

- Content-Type: `multipart/form-data`
- Body: PDF file with field name `file`

**Response:** `200 OK`

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Document processed successfully: nutrition-guide.pdf"
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| `400` | Invalid file type (only PDF accepted) |
| `500` | Processing failed |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

---

### Delete Document

Delete a document and all associated data (chunks, images, markdown).

```
DELETE /api/documents/{id}
```

**Response:** `200 OK`

```json
{
  "message": "Document deleted: 550e8400-e29b-41d4-a716-446655440000"
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| `404` | Document not found |
| `500` | Delete failed |

---

## Chunks

### Get Document Chunks

Retrieve all text chunks for a specific document.

```
GET /api/documents/{id}/chunks
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Maximum results (default: 100) |

**Response:** `200 OK`

```json
[
  {
    "id": "chunk-uuid-001",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "document_title": "Nutrition During Pregnancy",
    "text": "Good nutrition during pregnancy is essential for the health of both mother and baby...",
    "section_h1": "Why Nutrition Matters",
    "section_h2": null,
    "page_numbers": [1, 2],
    "chunk_index": 0,
    "token_count": 256
  }
]
```

---

### Get Single Chunk

Retrieve a specific chunk by ID.

```
GET /api/chunks/{id}
```

**Response:** `200 OK`

```json
{
  "id": "chunk-uuid-001",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_title": "Nutrition During Pregnancy",
  "text": "Good nutrition during pregnancy...",
  "section_h1": "Why Nutrition Matters",
  "section_h2": null,
  "page_numbers": [1, 2],
  "chunk_index": 0,
  "token_count": 256
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| `404` | Chunk not found |

---

## Images

### Get Document Images

Retrieve metadata for all images extracted from a document.

```
GET /api/documents/{id}/images
```

**Response:** `200 OK`

```json
[
  {
    "id": "image-uuid-001",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "page_number": 1,
    "image_index": 0,
    "file_path": "/data/images/550e8400.../page_001_img_000.png",
    "width": 800,
    "height": 600,
    "format": "png",
    "caption": null
  }
]
```

---

### Get Image Metadata

Retrieve metadata for a specific image.

```
GET /api/images/{id}
```

**Response:** `200 OK`

```json
{
  "id": "image-uuid-001",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "page_number": 1,
  "image_index": 0,
  "file_path": "/data/images/550e8400.../page_001_img_000.png",
  "width": 800,
  "height": 600,
  "format": "png",
  "caption": null
}
```

---

### Download Image File

Download the actual image file.

```
GET /api/images/{id}/file
```

**Response:** `200 OK`

- Content-Type: `image/png`, `image/jpeg`, etc.
- Body: Binary image data

**Example:**

```bash
# Download image
curl -o image.png "http://localhost:8000/api/images/image-uuid-001/file"
```

---

## Search

### Search Chunks

Search for relevant chunks using hybrid (vector + keyword), vector-only, or keyword-only search.

```
POST /api/search
```

**Request Body:**

```json
{
  "query": "What vitamins should I take during pregnancy?",
  "mode": "hybrid",
  "limit": 10,
  "title_filter": "nutrition"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query text |
| `mode` | string | No | Search mode: `hybrid` (default), `vector`, `keyword` |
| `limit` | integer | No | Max results: 1-100 (default: 10) |
| `title_filter` | string | No | Filter by document title (partial match) |

**Search Modes:**

| Mode | Description |
|------|-------------|
| `hybrid` | Combines vector similarity + BM25 keyword matching (recommended) |
| `vector` | Pure semantic similarity search |
| `keyword` | Pure BM25 keyword matching |

**Response:** `200 OK`

```json
[
  {
    "chunk_id": "chunk-uuid-001",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "document_title": "Nutrition During Pregnancy",
    "text": "Essential vitamins during pregnancy include folic acid, iron, calcium...",
    "page_numbers": [3, 4],
    "score": 0.89,
    "search_mode": "hybrid",
    "section_h1": "Vitamins and Minerals",
    "section_h2": "Prenatal Supplements"
  }
]
```

**Example:**

```bash
# Hybrid search
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "symptoms of gestational diabetes", "mode": "hybrid", "limit": 5}'

# Search with title filter
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "risk factors", "title_filter": "diabetes"}'
```

---

### Get LLM Context

Generate formatted context for LLM prompts with source attribution.

```
POST /api/search/context
```

**Request Body:**

```json
{
  "query": "What are the symptoms of preeclampsia?",
  "max_tokens": 4000,
  "mode": "hybrid"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Query for context retrieval |
| `max_tokens` | integer | No | Max tokens in context: 100-16000 (default: 4000) |
| `mode` | string | No | Search mode (default: `hybrid`) |

**Response:** `200 OK`

```json
{
  "context": "Documents referenced:\n- Hypertension in Pregnancy\n- Preeclampsia Guide\n\n---\n\n[Source: Hypertension in Pregnancy]\nPreeclampsia is characterized by high blood pressure and protein in urine...\n\n---\n\n[Source: Preeclampsia Guide]\nWarning signs include severe headaches, vision changes...",
  "documents_referenced": [
    "Hypertension in Pregnancy",
    "Preeclampsia Guide"
  ]
}
```

---

## Statistics

### Get Pipeline Statistics

Retrieve overall pipeline statistics.

```
GET /api/stats
```

**Response:** `200 OK`

```json
{
  "total_documents": 153,
  "total_chunks": 4521,
  "by_status": {
    "completed": 150,
    "failed": 2,
    "processing": 1
  }
}
```

---

## Health Check

### Check API Health

```
GET /health
```

**Response:** `200 OK`

```json
{
  "status": "healthy"
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

| Status | Description |
|--------|-------------|
| `200` | Success |
| `400` | Bad request (invalid parameters) |
| `404` | Resource not found |
| `500` | Internal server error |

---

## Rate Limits

No rate limits are currently enforced (demo mode).

---

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
