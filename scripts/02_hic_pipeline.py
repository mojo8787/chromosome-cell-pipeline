#!/usr/bin/env python3
"""
Hi-C Chromatin Analysis Pipeline.

Loads .mcool data, performs QC, and generates visualizations:
- Contact heatmap
- Distance decay (contacts vs genomic distance)
- QC stats

Uses cooltools and cooler (standard Hi-C format).

Usage:
    python scripts/02_hic_pipeline.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def find_cool_file(hic_dir):
    """Locate .mcool or .cool file (from cooltools download or manual)."""
    hic_dir = Path(hic_dir)
    for ext in ["*.mcool", "*.cool"]:
        for f in hic_dir.rglob(ext):
            return str(f)
    # cooltools may put test.mcool in data_dir root
    for p in [hic_dir / "test.mcool", ROOT / "data" / "test.mcool"]:
        if p.exists():
            return str(p)
    return None


def run_pipeline(config):
    import cooler
    import cooltools

    output_dir = Path(config["paths"]["output_dir"]) / "hic"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = config["hic"]["resolution"]
    cool_path = find_cool_file(config["paths"]["hic_dir"])

    if not cool_path:
        # Try cooltools download
        data_dir = Path(config["paths"]["hic_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            cool_path = cooltools.download_data(
                config["hic"]["download_dataset"],
                cache=True,
                data_dir=str(data_dir),
            )
        except Exception as e:
            print(f"Could not find or download Hi-C data: {e}")
            return None

    # Resolve path for mcool (multi-resolution)
    if ".mcool" in cool_path:
        uri = f"{cool_path}::resolutions/{resolution}"
    else:
        uri = cool_path

    print(f"Loading cooler: {uri}")
    clr = cooler.Cooler(uri)

    # QC stats
    chroms = clr.chromnames
    bins = clr.bins()[:]
    n_bins = bins.shape[0]
    try:
        n_pixels = int(clr.open("r")["pixels"].shape[0])
    except Exception:
        n_pixels = n_bins  # fallback
    sparsity = 1 - (n_pixels / (n_bins * n_bins)) if n_bins else 0

    qc_stats = {
        "chromosomes": chroms,
        "binsize": clr.binsize,
        "n_bins": n_bins,
        "n_pixels": n_pixels,
        "sparsity": sparsity,
    }
    print("QC:", qc_stats)

    # Save QC
    pd.DataFrame([qc_stats]).to_csv(output_dir / "qc_stats.csv", index=False)

    # Fetch matrix for first chromosome (or whole if small)
    chrom = chroms[0] if chroms else "chr1"
    mat = clr.matrix(balance=True).fetch(chrom)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Heatmap (Plotly)
    fig = go.Figure(
        data=go.Heatmap(
            z=np.log1p(mat),
            colorscale="Reds",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=f"Hi-C Contact Matrix — {chrom} @ {resolution / 1000:.0f} kb",
        xaxis_title="Genomic position",
        yaxis_title="Genomic position",
        height=500,
    )
    fig.write_html(output_dir / "heatmap.html")

    # Matplotlib fallback for PNG
    fig_mpl, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log1p(mat), cmap="Reds", aspect="equal")
    plt.colorbar(im, ax=ax, label="log(1 + contacts)")
    ax.set_title(f"Hi-C Contact Matrix — {chrom} @ {resolution / 1000:.0f} kb")
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("Genomic position")
    fig_mpl.savefig(output_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Distance decay: mean contacts per diagonal (genomic distance)
    try:
        n = mat.shape[0]
        diag_means = []
        diag_dists = []
        for k in range(1, min(n, 500)):  # limit for speed
            diag = np.diag(mat, k)
            if len(diag) > 0 and np.any(diag > 0):
                diag_means.append(np.nanmean(diag))
                diag_dists.append(k * clr.binsize)
        if diag_means and diag_dists:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=diag_dists,
                    y=diag_means,
                    mode="lines",
                    name="Contacts",
                )
            )
            fig2.update_layout(
                title="Contact frequency vs genomic distance",
                xaxis_title="Genomic distance (bp)",
                yaxis_title="Mean contact count",
                xaxis_type="log",
                yaxis_type="log",
                height=400,
            )
            fig2.write_html(output_dir / "distance_decay.html")
    except Exception as e:
        print(f"Distance decay skipped: {e}")

    print(f"Outputs saved to {output_dir}")
    return output_dir


def main():
    config = load_config()
    os.chdir(ROOT)
    run_pipeline(config)


if __name__ == "__main__":
    main()
