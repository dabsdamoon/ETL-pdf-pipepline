#!/usr/bin/env python3
"""Simple HTTP server for the demo frontend."""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 3000
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


def main():
    os.chdir(FRONTEND_DIR)

    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Frontend server running at http://localhost:{PORT}")
        print(f"Serving files from: {FRONTEND_DIR}")
        print("\nMake sure the API server is running on http://localhost:8000")
        print("Press Ctrl+C to stop\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
