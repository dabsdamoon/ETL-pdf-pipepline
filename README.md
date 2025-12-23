# ETL PDF Pipeline

A comprehensive ETL pipeline for extracting, transforming, and loading PDF documents into vector databases for RAG (Retrieval-Augmented Generation) applications.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/dabsdamoon/ETL-pdf-pipepline.git@main

# Or install locally for development
pip install -e /path/to/ETL-pdf-pipeline
```

## Quick Start

```python
from etl_pdf_pipeline import extract_pdf, chunk_text, embed_chunks

# Extract PDF to markdown
markdown, metadata = extract_pdf("document.pdf")

# Chunk the text
chunks = chunk_text(markdown, document_id=metadata["document_id"], title=metadata["title"])

# Generate embeddings
embedded_chunks = embed_chunks(chunks)
```

## Extraction Methods

### PyMuPDF (Default)

Uses PyMuPDF/pymupdf4llm for fast text extraction. Best for digital PDFs with selectable text.

```python
from etl_pdf_pipeline import extract_pdf

markdown, metadata = extract_pdf("document.pdf")
```

### Google Vision OCR

Uses Google Cloud Vision API for OCR extraction. Best for scanned PDFs or documents with complex layouts.

#### Setup

1. **Create a Google Cloud Project** and enable the Vision API:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the [Cloud Vision API](https://console.cloud.google.com/apis/library/vision.googleapis.com)

2. **Set up authentication** (choose one method):

   **Option A: Service Account (Recommended for production)**
   ```bash
   # Create a service account and download the JSON key
   # Then set the environment variable:
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

   **Option B: Application Default Credentials (for local development)**
   ```bash
   # Login with gcloud CLI
   gcloud auth application-default login
   ```

3. **Install the required package:**
   ```bash
   pip install google-cloud-vision
   ```

#### Usage

```python
from etl_pdf_pipeline import extract_pdf, Config, ExtractionConfig

# Configure Google Vision OCR extraction
config = Config(
    extraction=ExtractionConfig(method="google_vision")
)

# Extract with OCR
markdown, metadata = extract_pdf("scanned_document.pdf", config=config)
print(metadata["extraction_method"])  # "google_vision"
```

## Configuration

### Full Configuration Example

```python
from etl_pdf_pipeline import Config, ExtractionConfig, ChunkingConfig, EmbeddingConfig

config = Config(
    extraction=ExtractionConfig(
        method="pymupdf",      # "pymupdf" or "google_vision"
        ocr_dpi=300,           # DPI for OCR page rendering
    ),
    chunking=ChunkingConfig(
        chunk_size=512,        # Tokens per chunk
        chunk_overlap=50,      # Overlap between chunks
    ),
    embedding=EmbeddingConfig(
        provider="openai",     # Only "openai" is supported
        openai_model="text-embedding-3-small",
    ),
)
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud service account JSON | Yes (if using Google Vision OCR) |

## API Reference

### `extract_pdf(pdf_path, document_id=None, config=None)`

Extract text from a PDF file to markdown.

**Parameters:**
- `pdf_path` (str): Path to the PDF file
- `document_id` (str, optional): Unique document identifier
- `config` (Config, optional): Configuration object

**Returns:**
- `tuple[str, dict]`: (markdown_content, metadata)

### `chunk_text(text, document_id=None, title="", config=None)`

Chunk text content using hybrid markdown-aware chunking.

**Parameters:**
- `text` (str): Text or markdown content to chunk
- `document_id` (str, optional): Document identifier
- `title` (str, optional): Document title
- `config` (Config, optional): Configuration object

**Returns:**
- `list[Chunk]`: List of Chunk objects

### `embed_chunks(chunks, config=None)`

Generate embeddings for chunks.

**Parameters:**
- `chunks` (list[Chunk]): List of Chunk objects
- `config` (Config, optional): Configuration object

**Returns:**
- `list[Chunk]`: Chunks with embeddings populated

### `process_pdf(pdf_path, config=None)`

Full ETL pipeline: extract, chunk, and embed a PDF.

**Parameters:**
- `pdf_path` (str): Path to the PDF file
- `config` (Config, optional): Configuration object

**Returns:**
- `tuple[list[Chunk], dict]`: (embedded_chunks, metadata)

## License

MIT
