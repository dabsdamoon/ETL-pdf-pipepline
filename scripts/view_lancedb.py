#!/usr/bin/env python3
"""View LanceDB contents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from src.config import Config


def main():
    config = Config()
    db_path = config.paths.lancedb_dir

    if not db_path.exists():
        print(f"LanceDB not found at: {db_path}")
        return

    db = lancedb.connect(str(db_path))
    tables = db.table_names()

    print(f"Database: {db_path}")
    print(f"Tables: {tables}\n")

    for table_name in tables:
        table = db.open_table(table_name)
        count = table.count_rows()
        print(f"=== {table_name} ({count} rows) ===")

        # Show schema
        schema = table.schema
        print(f"Columns: {schema.names}\n")

        # Show sample data (without vector column for readability)
        df = table.to_pandas()
        display_cols = [c for c in df.columns if c != 'vector']

        if len(df) > 0:
            print(df[display_cols].head(10).to_string())
        else:
            print("(empty)")
        print()


if __name__ == "__main__":
    main()
