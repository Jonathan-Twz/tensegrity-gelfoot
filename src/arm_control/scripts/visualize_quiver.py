#!/usr/bin/env python3
"""Visualize a saved shear-field .npy file as a quiver plot."""

import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_ROOT.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

def plot_vector_field(ax, vf, ch_dim=0, color='blue', scale=6, title=''):
    if isinstance(vf, np.ndarray):
        data = vf
    else:
        data = np.asarray(vf)
    if ch_dim == 0:
        u = data[0]
        v = data[1]
    elif ch_dim == 2:
        u = data[..., 0]
        v = data[..., 1]
    else:
        raise ValueError('Unsupported channel dimension for vector field')

    field = ax.quiver(u, v, color=color, angles='xy', scale_units='xy', scale=scale)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title)
    return field

def load_shear_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 1:
        size = int(round((data.size / 2) ** 0.5))
        data = data.reshape(2, size, size)
    elif data.ndim == 3 and data.shape[0] != 2 and data.shape[-1] == 2:
        data = np.moveaxis(data, -1, 0)
    if data.shape[0] != 2:
        raise ValueError(f"Expected 2 channels for shear vector field, got shape {data.shape}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npy_path', type=Path, help='Path to shear field numpy file (e.g. ~/gelfoot_dataset/shear_fields/000057.npy)')
    parser.add_argument('--scale', type=float, default=6.0, help='Arrow scaling factor for quiver plot')
    parser.add_argument('--color', default='tab:red', help='Color for quiver arrows')
    args = parser.parse_args()

    shear_field = load_shear_array(args.npy_path.expanduser().resolve())

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_vector_field(ax, shear_field, ch_dim=0, color=args.color, scale=args.scale, title=args.npy_path.name)
    fig.suptitle(str(args.npy_path))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
