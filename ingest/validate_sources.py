"""Validate corpus/sources.yaml against the P0-3 schema.

Usage: uv run python ingest/validate_sources.py [path/to/sources.yaml]
"""

import sys
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-_]*$")
    url: str
    doc_type: str
    instrument_family: str
    vendor: str
    version: str
    date_fetched: date
    license_note: str


def load_sources(path: str = "corpus/sources.yaml") -> list[Source]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Source(**row) for row in data["sources"]]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/sources.yaml"
    try:
        sources = load_sources(path)
    except Exception as e:
        print(f"INVALID: {path} — {e}")
        return 1
    print(f"OK: {len(sources)} source(s) in {path}")
    for s in sources:
        print(f"  - {s.id}: {s.url} ({s.license_note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
