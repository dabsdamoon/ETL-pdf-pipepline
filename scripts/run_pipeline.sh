#!/bin/bash
# ETL Pipeline Runner
# Usage: ./scripts/run.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate etl-pdf-pipeline 2>/dev/null || true
fi

# Default command is 'process'
COMMAND="${1:-help}"

case "$COMMAND" in
    process)
        shift
        python scripts/run_pipeline.py process "$@"
        ;;
    search)
        shift
        python scripts/run_pipeline.py search "$@"
        ;;
    reprocess)
        shift
        python scripts/run_pipeline.py reprocess "$@"
        ;;
    delete)
        shift
        python scripts/run_pipeline.py delete "$@"
        ;;
    stats)
        python scripts/run_pipeline.py stats
        ;;
    list)
        shift
        python scripts/run_pipeline.py list "$@"
        ;;
    help|--help|-h)
        echo "ETL Pipeline for PDF Document Processing"
        echo ""
        echo "Usage: ./scripts/run.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  process              Process PDF documents"
        echo "    --file, -f         Process a single PDF file"
        echo "    --directory, -d    Process all PDFs in a directory"
        echo "    --incremental, -i  Only process new/changed documents"
        echo ""
        echo "  search <query>       Search processed documents"
        echo "    --mode, -m         Search mode: vector, hybrid, keyword (default: hybrid)"
        echo "    --limit, -n        Number of results (default: 5)"
        echo "    --title-filter, -t Filter by document title"
        echo ""
        echo "  reprocess            Reprocess document from markdown"
        echo "    --document-id, -d  Document ID to reprocess"
        echo ""
        echo "  delete               Delete a document"
        echo "    --document-id, -d  Document ID to delete"
        echo ""
        echo "  stats                Show pipeline statistics"
        echo "  list                 List documents"
        echo "    --status, -s       Filter by status: pending, processing, completed, failed"
        echo ""
        echo "Examples:"
        echo "  ./scripts/run.sh process --incremental"
        echo "  ./scripts/run.sh process --file path/to/doc.pdf"
        echo "  ./scripts/run.sh search \"What are the symptoms?\""
        echo "  ./scripts/run.sh stats"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run './scripts/run.sh help' for usage"
        exit 1
        ;;
esac
