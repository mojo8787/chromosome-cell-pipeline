#!/usr/bin/env python3
"""
Download data for the Chromosome + Cell mini-pipeline.

1. HFF Micro-C Hi-C data via cooltools.download_data
2. BBBC039 nuclei microscopy images from Broad Institute

Usage:
    python scripts/01_download_data.py
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def download_hic(config):
    """Download HFF Micro-C Hi-C data via cooltools."""
    hic_dir = Path(config["paths"]["hic_dir"])
    hic_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cooltools
        dataset = config["hic"]["download_dataset"]
        print(f"Downloading {dataset} via cooltools...")
        cool_file = cooltools.download_data(dataset, cache=True, data_dir=str(hic_dir))
        print(f"  -> {cool_file}")
        return True
    except Exception as e:
        print(f"  cooltools download failed: {e}")
        print("  Fallback: You can manually download from 4D Nucleome or open2c examples.")
        return False


def download_bbbc039(config):
    """Download BBBC039 nuclei images from Broad Institute."""
    microscopy_dir = Path(config["paths"]["microscopy_dir"])
    microscopy_dir.mkdir(parents=True, exist_ok=True)
    images_dir = microscopy_dir / "images"
    url = config["microscopy"]["bbbc039_url"]
    zip_path = microscopy_dir / "images.zip"

    # Check if we already have images (any .tif in microscopy dir)
    if images_dir.exists() and list(images_dir.glob("*.tif*")):
        print("BBBC039 images already present.")
        return True

    if not zip_path.exists():
        print("Downloading BBBC039 images (~78 MB)...")
        try:
            urlretrieve(url, zip_path)
            print(f"  -> {zip_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return False

    print("Extracting BBBC039...")
    try:
        import shutil
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(microscopy_dir)
        # Collect all .tif files (may be in subdir or root)
        images_dir.mkdir(exist_ok=True)
        for f in microscopy_dir.rglob("*.tif*"):
            if f.is_file():
                dest = images_dir / f.name
                # Skip if already in images_dir (zip may extract directly there)
                if dest.resolve() != f.resolve():
                    shutil.copy2(f, dest)
        # Remove subdirs that are not images (avoid duplicates)
        for d in list(microscopy_dir.iterdir()):
            if d.is_dir() and d.name != "images" and d != images_dir:
                shutil.rmtree(d, ignore_errors=True)
        print("  -> Extracted to", images_dir)
        return True
    except Exception as e:
        print(f"  Extract failed: {e}")
        return False


def main():
    config = load_config()
    os.chdir(ROOT)

    ok_hic = download_hic(config)
    ok_micro = download_bbbc039(config)

    if ok_hic and ok_micro:
        print("\nAll data downloaded successfully.")
    else:
        print("\nSome downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
