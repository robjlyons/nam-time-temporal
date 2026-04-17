#!/usr/bin/env python3
"""
Print key metadata from a .nam model file.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("model_path", help="Path to .nam file")
    args = parser.parse_args()

    p = Path(args.model_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("r") as fp:
        d = json.load(fp)

    st = p.stat()
    modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    print(json.dumps(
        {
            "path": str(p),
            "version": d.get("version"),
            "architecture": d.get("architecture"),
            "config": d.get("config"),
            "modified_utc": modified,
            "size_bytes": st.st_size,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
