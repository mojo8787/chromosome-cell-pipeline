#!/usr/bin/env python3
"""
Batch run: execute download + both pipelines to generate all outputs.

Usage:
    python scripts/04_generate_outputs.py
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def run(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
    )
    return result.returncode == 0


def main():
    print("1. Downloading data...")
    if not run("scripts/01_download_data.py"):
        print("Download failed. Exiting.")
        sys.exit(1)

    print("\n2. Running Hi-C pipeline...")
    run("scripts/02_hic_pipeline.py")

    print("\n3. Running microscopy pipeline...")
    run("scripts/03_microscopy_pipeline.py")

    # Optional: VLM analysis (requires API key)
    api_key_env = "OPENAI_API_KEY"
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        api_key_env = cfg.get("vlm", {}).get("api_key_env", "OPENAI_API_KEY")
    if os.environ.get(api_key_env):
        print("\n4. Running VLM analysis...")
        run("scripts/05_vlm_analysis.py")
    else:
        print("\n4. Skipping VLM analysis (no API key). Run manually if needed.")

    print("\nDone. Run `streamlit run app.py` to view the dashboard.")


if __name__ == "__main__":
    main()
