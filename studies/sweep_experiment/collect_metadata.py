#!/usr/bin/env python3
"""Aggregate per-image JSON metadata sidecars into a single dataset-level JSON."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: collect_metadata.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    records = []

    for sample_dir in sorted(dataset_dir.glob("disk_*")):
        sidecars = list(sample_dir.glob("disk_*.json"))
        if not sidecars:
            continue
        with open(sidecars[0], "r") as f:
            records.append(json.load(f))

    records.sort(key=lambda r: r["name"])

    out_path = dataset_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump({"dataset_dir": str(dataset_dir), "n_images": len(records), "images": records}, f, indent=2)

    print(f"Collected metadata for {len(records)} images -> {out_path}")


if __name__ == "__main__":
    main()
