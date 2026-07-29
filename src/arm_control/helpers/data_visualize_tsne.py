#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE of 2×30×30 arrays (from per-sample .npy files listed in a CSV index),
with a 2×3 subplot grid colored by the 6D wrench: [Fx, Fy, Fz, Tx, Ty, Tz].

Usage example:
  python tsne_wrench_grid.py \
    --index_csv /path/to/index.csv \
    --standardize \
    --pca 50 \
    --perplexity 30 \
    --lr 200 \
    --iter 1500 \
    --out_prefix shear_tsne \
    --save_individual
"""

import argparse, ast, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------
# Helpers
# ---------------------------

def parse_wrench(x):
    """CSV has a string like '[-0.89, 0.62, ...]'; turn it into np.ndarray(6,)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        arr = np.asarray(ast.literal_eval(x), dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"wrench must be length-6, got shape {arr.shape}")
    return arr

def load_from_index(index_csv: Path, root: Path):
    df = pd.read_csv(index_csv)
    required = {"wrench", "array_path"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV must include columns {required}; missing {missing_cols}")

    X_list, Y_list, meta = [], [], []
    missing_files = 0

    for _, row in df.iterrows():
        rel_path = str(row["array_path"]).strip()
        fpath = (root / rel_path).resolve()
        if not fpath.exists():
            warnings.warn(f"Missing array file: {fpath}")
            missing_files += 1
            continue

        arr = np.load(fpath)  # expected (2,30,30)
        if arr.shape != (2, 30, 30):
            warnings.warn(f"Unexpected shape {arr.shape} in {fpath}, skipping")
            continue

        X_list.append(arr.reshape(-1).astype(np.float32))  # -> (1800,)
        Y_list.append(parse_wrench(row["wrench"]))
        meta.append((
            row.get("idx", None),
            row.get("timestamp_nsec", None),
            rel_path
        ))

    if not X_list:
        raise RuntimeError("No valid arrays loaded. Check paths/shapes in CSV.")

    X = np.stack(X_list, axis=0)  # (N, 1800)
    Y = np.stack(Y_list, axis=0)  # (N, 6)
    return X, Y, meta, missing_files

def run_tsne(X, pca_dim, perplexity, lr, n_iter, random_state, do_standardize):
    # Standardize (helps PCA/t-SNE); optional
    if do_standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # Optional PCA for speed/stability
    Xfeat = X
    if pca_dim and 0 < pca_dim < X.shape[1]:
        ncomp = min(pca_dim, X.shape[1])
        pca = PCA(n_components=ncomp, random_state=random_state)
        Xfeat = pca.fit_transform(X)
        print(f"[Info] PCA -> {Xfeat.shape[1]} dims (explained var: {pca.explained_variance_ratio_.sum():.3f})")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=lr,
        n_iter=n_iter,
        init="pca",
        metric="euclidean",
        random_state=random_state,
        verbose=1,
    )
    X2 = tsne.fit_transform(Xfeat)  # (N,2)
    return X2

def save_embedding_csv(out_path, X2, meta, Y):
    rows = []
    for (m, emb, w) in zip(meta, X2, Y):
        idx, ts, rel = m
        rows.append({
            "idx": idx,
            "timestamp_nsec": ts,
            "array_path": rel,
            "x": emb[0],
            "y": emb[1],
            "Fx": w[0], "Fy": w[1], "Fz": w[2],
            "Tx": w[3], "Ty": w[4], "Tz": w[5],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Saved embedding CSV -> {out_path}")

def plot_wrench_grid(X2, Y, out_png, title_prefix="t-SNE of 2×30×30 arrays", save_individual=False, indiv_prefix="tsne_comp"):
    """
    Draw a 2×3 grid for (Fx,Fy,Fz,Tx,Ty,Tz).
    Also optionally save six standalone PNGs.
    """
    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]  # column order 0..5
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=Y[:, i], s=10)
        ax.set_title(labels[i])
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(labels[i])

        if save_individual:
            plt.figure(figsize=(6, 5))
            sc2 = plt.scatter(X2[:, 0], X2[:, 1], c=Y[:, i], s=10)
            plt.xlabel("t-SNE dim 1")
            plt.ylabel("t-SNE dim 2")
            plt.title(f"{title_prefix} — {labels[i]}")
            cbar = plt.colorbar(sc2)
            cbar.set_label(labels[i])
            plt.tight_layout()
            fname = f"{indiv_prefix}_{labels[i]}.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"[OK] Saved {labels[i]} -> {fname}")

    fig.suptitle(f"{title_prefix} (colored by wrench components)", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved 2×3 grid -> {out_png}")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="t-SNE + 2×3 wrench component visualization")
    ap.add_argument("--index_csv", required=True, help="CSV with columns: idx,timestamp_nsec,wrench,array_path")
    ap.add_argument("--root", default="", help="Root dir for array_path; defaults to CSV's directory")
    ap.add_argument("--subsample", type=int, default=0, help="Keep every k-th sample (k>1) for speed")
    ap.add_argument("--standardize", action="store_true", help="Z-score features before PCA/t-SNE")
    ap.add_argument("--pca", type=int, default=50, help="PCA dims before t-SNE (0 to disable)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=200.0, help="t-SNE learning rate")
    ap.add_argument("--iter", type=int, default=1000, help="t-SNE iterations")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_prefix", default="tsne_wrench", help="Output prefix (PNG & CSV)")
    ap.add_argument("--save_individual", action="store_true", help="Also save 6 separate component PNGs")
    args = ap.parse_args()

    index_csv = Path(args.index_csv).resolve()
    root = Path(args.root).resolve() if args.root else index_csv.parent

    # Load
    X, Y, meta, missing = load_from_index(index_csv, root)
    N = X.shape[0]
    print(f"[Info] Loaded {N} samples (skipped missing files: {missing})")

    # Optional subsample
    if args.subsample and args.subsample > 1:
        X = X[::args.subsample]
        Y = Y[::args.subsample]
        meta = meta[::args.subsample]
        N = X.shape[0]
        print(f"[Info] After subsample(k={args.subsample}): {N} samples")

    # t-SNE (run once)
    X2 = run_tsne(
        X,
        pca_dim=args.pca,
        perplexity=args.perplexity,
        lr=args.lr,
        n_iter=args.iter,
        random_state=args.random_state,
        do_standardize=args.standardize,
    )

    # Save embedding with metadata & wrench
    embed_csv = f"{args.out_prefix}_embed.csv"
    save_embedding_csv(embed_csv, X2, meta, Y)

    # 2×3 grid colored by Fx..Tz
    grid_png = f"{args.out_prefix}_grid.png"
    plot_wrench_grid(
        X2, Y, grid_png,
        title_prefix=f"t-SNE (N={N})",
        save_individual=args.save_individual,
        indiv_prefix=f"{args.out_prefix}_comp"
    )

if __name__ == "__main__":
    main()
