"""POST the sample tickets to a running service: python -m scripts.seed [--url ...] [--file ...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "samples" / "tickets.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the ticket service with sample tickets.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args(argv)

    tickets = json.loads(args.file.read_text(encoding="utf-8"))
    ok = True
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for ticket in tickets:
            try:
                status = client.post("/tickets", json=ticket).status_code
            except httpx.HTTPError as exc:
                print(f"{ticket['id']} error: {exc}", file=sys.stderr)
                ok = False
                continue
            print(f"{ticket['id']} {status}")
            ok = ok and status in (200, 202)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
