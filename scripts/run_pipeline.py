#!/usr/bin/env python3
"""CLI entry point for the ETL pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import Pipeline
from src.logging_config import setup_logging, logger
from src.retrieve import HybridRetriever, SearchMode


def main():
    parser = argparse.ArgumentParser(
        description="ETL Pipeline for PDF Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in the default directory
  python run_pipeline.py process

  # Process a single PDF
  python run_pipeline.py process --file path/to/document.pdf

  # Process only new documents (incremental)
  python run_pipeline.py process --incremental

  # Reprocess a document from markdown
  python run_pipeline.py reprocess --document-id <id>

  # Search the processed documents
  python run_pipeline.py search "What are the symptoms of diabetes?"

  # Show pipeline statistics
  python run_pipeline.py stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDF documents")
    process_parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Process a single PDF file",
    )
    process_parser.add_argument(
        "--directory", "-d",
        type=Path,
        help="Process all PDFs in a directory",
    )
    process_parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="Only process new or changed documents",
    )
    process_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    # Reprocess command
    reprocess_parser = subparsers.add_parser(
        "reprocess",
        help="Reprocess a document from markdown",
    )
    reprocess_parser.add_argument(
        "--document-id", "-d",
        required=True,
        help="Document ID to reprocess",
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument(
        "--document-id", "-d",
        required=True,
        help="Document ID to delete",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search processed documents")
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--mode", "-m",
        choices=["vector", "hybrid", "keyword"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    search_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    search_parser.add_argument(
        "--title-filter", "-t",
        help="Filter by document title",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show pipeline statistics")

    # List command
    list_parser = subparsers.add_parser("list", help="List documents")
    list_parser.add_argument(
        "--status", "-s",
        choices=["pending", "processing", "completed", "failed"],
        help="Filter by status",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set up logging
    config = Config()
    setup_logging(config.paths.logs_dir)

    try:
        if args.command == "process":
            return cmd_process(args, config)
        elif args.command == "reprocess":
            return cmd_reprocess(args, config)
        elif args.command == "delete":
            return cmd_delete(args, config)
        elif args.command == "search":
            return cmd_search(args, config)
        elif args.command == "stats":
            return cmd_stats(config)
        elif args.command == "list":
            return cmd_list(args, config)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def cmd_process(args, config: Config) -> int:
    """Process PDF documents."""
    pipeline = Pipeline(config)

    try:
        if args.file:
            # Process single file
            print(f"Processing: {args.file}")
            doc_id = pipeline.process_document(args.file)
            print(f"Done. Document ID: {doc_id}")

        elif args.incremental:
            # Incremental processing
            print("Processing new/changed documents...")
            doc_ids = pipeline.process_new_documents(progress=not args.no_progress)
            print(f"Done. Processed {len(doc_ids)} documents.")

        else:
            # Process directory
            directory = args.directory or config.paths.pdf_dir
            print(f"Processing all PDFs in: {directory}")
            doc_ids = pipeline.process_directory(directory, progress=not args.no_progress)
            print(f"Done. Processed {len(doc_ids)} documents.")

        return 0

    finally:
        pipeline.close()


def cmd_reprocess(args, config: Config) -> int:
    """Reprocess a document from markdown."""
    pipeline = Pipeline(config)

    try:
        print(f"Reprocessing document: {args.document_id}")
        pipeline.reprocess_from_markdown(args.document_id)
        print("Done.")
        return 0

    finally:
        pipeline.close()


def cmd_delete(args, config: Config) -> int:
    """Delete a document."""
    pipeline = Pipeline(config)

    try:
        print(f"Deleting document: {args.document_id}")
        pipeline.delete_document(args.document_id)
        print("Done.")
        return 0

    finally:
        pipeline.close()


def cmd_search(args, config: Config) -> int:
    """Search processed documents."""
    retriever = HybridRetriever(config)

    mode_map = {
        "vector": SearchMode.VECTOR,
        "hybrid": SearchMode.HYBRID,
        "keyword": SearchMode.KEYWORD,
    }
    mode = mode_map[args.mode]

    print(f"\nSearching ({args.mode}): {args.query}\n")
    print("-" * 60)

    results = retriever.search(
        args.query,
        mode=mode,
        limit=args.limit,
        title_filter=args.title_filter,
    )

    if not results:
        print("No results found.")
        return 0

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.document_title}")
        print(f"    Score: {r.score:.4f}")
        if r.section_h1:
            print(f"    Section: {r.section_h1}")
        print(f"    Text: {r.text[:200]}...")

    print("\n" + "-" * 60)
    return 0


def cmd_stats(config: Config) -> int:
    """Show pipeline statistics."""
    pipeline = Pipeline(config)

    try:
        stats = pipeline.get_stats()

        print("\nPipeline Statistics")
        print("=" * 40)
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total chunks:    {stats['total_chunks']}")
        print("\nBy status:")
        for status, count in stats["by_status"].items():
            print(f"  {status}: {count}")

        return 0

    finally:
        pipeline.close()


def cmd_list(args, config: Config) -> int:
    """List documents."""
    from src.models import DocumentStatus
    from src.load import SQLiteStore

    store = SQLiteStore(config)

    try:
        status = DocumentStatus(args.status) if args.status else None
        documents = store.list_documents(status)

        if not documents:
            print("No documents found.")
            return 0

        print(f"\n{'ID':<36} {'Status':<12} {'Title'}")
        print("-" * 80)

        for doc in documents:
            title = doc.title[:35] + "..." if len(doc.title) > 38 else doc.title
            print(f"{doc.id} {doc.status.value:<12} {title}")

        print(f"\nTotal: {len(documents)} documents")
        return 0

    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
