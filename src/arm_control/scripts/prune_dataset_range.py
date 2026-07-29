#!/usr/bin/env python3
"""Remove a range of dataset samples and renumber subsequent entries."""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

START_IDX = 14905
END_IDX = 21634
DIGITS = 6

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path.home() / "gelfoot_dataset",
        help="Directory containing index.csv and shear_fields/",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=START_IDX,
        help="First index (inclusive) to remove.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=END_IDX,
        help="Last index (inclusive) to remove.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying files.",
    )
    return parser.parse_args()

def load_index(index_path: Path) -> List[Dict[str, str]]:
    with index_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)

def save_index(index_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = rows[0].keys()
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def format_array_path(relative_path: str, new_idx: int) -> str:
    rel = Path(relative_path)
    filename = f"{new_idx:0{DIGITS}d}{rel.suffix}"
    return str(rel.with_name(filename))

def delete_file(path: Path) -> None:
    if path.exists():
        path.unlink()

def rename_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)

def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    index_path = dataset_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found at {index_path}")

    rows = load_index(index_path)
    start = args.start
    end = args.end
    if start > end:
        raise ValueError("start index must be <= end index")
    removed = end - start + 1

    kept_rows: List[Dict[str, str]] = []
    rename_ops = []
    delete_paths = []

    for row in rows:
        try:
            idx = int(row["idx"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid idx in row: {row}") from exc

        array_rel = row.get("array_path")
        array_path = dataset_dir / array_rel if array_rel else None

        if start <= idx <= end:
            if array_path is not None:
                delete_paths.append(array_path)
            continue

        if idx > end:
            new_idx = idx - removed
            row = dict(row)
            row["idx"] = str(new_idx)
            if array_path is not None:
                new_rel = format_array_path(array_rel, new_idx)
                new_path = dataset_dir / new_rel
                rename_ops.append((array_path, new_path))
                row["array_path"] = new_rel
        kept_rows.append(row)

    kept_rows.sort(key=lambda r: int(r["idx"]))

    if args.dry_run:
        print(f"Dataset directory: {dataset_dir}")
        print(f"Removing indices [{start}, {end}] (count={removed})")
        print("Files to delete:")
        for path in delete_paths:
            print(f"  DEL {path}")
        print("Files to rename:")
        for src, dst in rename_ops:
            print(f"  MV {src} -> {dst}")
        print(f"New row count: {len(kept_rows)}")
        return

    for path in delete_paths:
        delete_file(path)

    for src, dst in rename_ops:
        rename_file(src, dst)

    save_index(index_path, kept_rows)
    print(f"Removed indices [{start}, {end}] and renumbered subsequent entries.")

if __name__ == "__main__":
    main()
