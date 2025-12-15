#!/usr/bin/env python3
"""Run the ETL Pipeline API server."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
    )


if __name__ == "__main__":
    main()
