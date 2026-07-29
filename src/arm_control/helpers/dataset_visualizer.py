#!/usr/bin/env python3
"""Visualize a random shear field sample from a recorded dataset."""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gelslim_shear.plot_utils.shear_plotter import plot_vector_field


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path.home() / "gelfoot_dataset",
        help="Directory containing index.csv and shear field files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=6.0,
        help="Arrow scaling factor passed to plot_vector_field.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def load_index(index_path: Path):
    entries = []
    with index_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("array_path"):
                continue
            entries.append(row)
    return entries


def reshape_shear(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 3:
        if data.shape[0] in (2, 3, 4, 5):
            return data
        if data.shape[-1] in (2, 3, 4, 5):
            return np.moveaxis(data, -1, 0)
    if data.ndim == 1:
        for channels in (2, 3, 4, 5):
            if data.size % channels:
                continue
            side = int(round((data.size / channels) ** 0.5))
            if side * side * channels == data.size:
                return data.reshape((channels, side, side))
    raise ValueError(f"Cannot infer shear field shape from data with shape {data.shape}")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    index_path = dataset_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    entries = load_index(index_path)
    if not entries:
        raise RuntimeError(f"No entries with shear field paths in {index_path}")

    sample = random.choice(entries)
    shear_path = dataset_dir / sample["array_path"]
    shear_data = np.load(shear_path)
    shear_field = reshape_shear(shear_data)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_vector_field(ax, shear_field, ch_dim=0, color="tab:red", scale=args.scale, title=shear_path.name)
    fig.suptitle(f"Sample {sample['idx']} | timestamp {sample['timestamp_nsec']}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
