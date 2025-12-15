#!/usr/bin/env python3
"""Utility script to reprocess documents from markdown checkpoints."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import Pipeline
from src.logging_config import setup_logging
from src.load import SQLiteStore


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess documents from markdown checkpoints",
        epilog="""
This utility re-transforms documents from their markdown checkpoint files.
Useful when you want to:
- Change chunking parameters
- Update embeddings
- Fix chunking issues without re-extracting from PDFs

Examples:
  # Reprocess a single document
  python reprocess_markdown.py --document-id <id>

  # Reprocess all documents
  python reprocess_markdown.py --all

  # Reprocess failed documents
  python reprocess_markdown.py --failed
        """,
    )

    parser.add_argument(
        "--document-id", "-d",
        help="Reprocess a specific document by ID",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Reprocess all documents",
    )
    parser.add_argument(
        "--failed", "-f",
        action="store_true",
        help="Reprocess only failed documents",
    )

    args = parser.parse_args()

    if not any([args.document_id, args.all, args.failed]):
        parser.print_help()
        return 1

    config = Config()
    setup_logging(config.paths.logs_dir)

    pipeline = Pipeline(config)
    store = SQLiteStore(config)

    try:
        if args.document_id:
            # Single document
            print(f"Reprocessing document: {args.document_id}")
            pipeline.reprocess_from_markdown(args.document_id)
            print("Done.")

        elif args.all:
            # All documents
            from src.models import DocumentStatus
            documents = store.list_documents(DocumentStatus.COMPLETED)
            print(f"Reprocessing {len(documents)} documents...")

            for i, doc in enumerate(documents, 1):
                print(f"[{i}/{len(documents)}] {doc.title}")
                try:
                    pipeline.reprocess_from_markdown(doc.id)
                except Exception as e:
                    print(f"  Failed: {e}")

            # Rebuild FTS index
            pipeline.lancedb_store.create_fts_index()
            print("Done.")

        elif args.failed:
            # Failed documents
            from src.models import DocumentStatus
            documents = store.list_documents(DocumentStatus.FAILED)

            if not documents:
                print("No failed documents found.")
                return 0

            print(f"Reprocessing {len(documents)} failed documents...")

            for i, doc in enumerate(documents, 1):
                print(f"[{i}/{len(documents)}] {doc.title}")
                try:
                    pipeline.reprocess_from_markdown(doc.id)
                except Exception as e:
                    print(f"  Failed: {e}")

            print("Done.")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    finally:
        pipeline.close()
        store.close()


if __name__ == "__main__":
    sys.exit(main())
