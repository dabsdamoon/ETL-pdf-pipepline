#!/usr/bin/env python3
"""Utility to flush/reset pipeline data."""

import argparse
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Flush ETL pipeline data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Flush all processed data (keep source PDFs)
  python scripts/flush_data.py

  # Flush everything including source PDFs
  python scripts/flush_data.py --all

  # Dry run - show what would be deleted
  python scripts/flush_data.py --dry-run
        """,
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Also delete source PDFs",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()
    config = Config()

    # Directories to delete
    dirs_to_delete = [
        ("SQLite DB", config.paths.sqlite_path.parent),
        ("LanceDB", config.paths.lancedb_dir),
        ("Markdown", config.paths.markdown_dir),
        ("Images", config.paths.images_dir),
        ("Logs", config.paths.logs_dir),
    ]

    if args.all:
        dirs_to_delete.append(("Source PDFs", config.paths.pdf_dir))

    # Show what will be deleted
    print("The following will be deleted:")
    print("-" * 40)

    total_size = 0
    for name, path in dirs_to_delete:
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            total_size += size
            size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.2f} KB"
            print(f"  {name}: {path} ({size_str})")
        else:
            print(f"  {name}: {path} (not found)")

    print("-" * 40)
    print(f"Total: {total_size / 1024 / 1024:.2f} MB")
    print()

    if args.dry_run:
        print("Dry run - nothing deleted.")
        return 0

    # Confirm
    if not args.yes:
        response = input("Are you sure you want to delete? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            return 0

    # Delete
    for name, path in dirs_to_delete:
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            print(f"Deleted: {path}")

    print("\nData flushed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
