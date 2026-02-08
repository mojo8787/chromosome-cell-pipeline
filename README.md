# Chromosome + Cell Mini-Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18500690.svg)](https://doi.org/10.5281/zenodo.18500690)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/v/release/mojo8787/chromosome-cell-pipeline)](https://github.com/mojo8787/chromosome-cell-pipeline/releases)
[![CI](https://github.com/mojo8787/chromosome-cell-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/mojo8787/chromosome-cell-pipeline/actions/workflows/ci.yml)

A combined genomics and microscopy analysis pipeline demonstrating end-to-end Hi-C chromatin analysis and nuclei segmentation. Inspired by chromosome organization workflows; designed as a reproducible demo pipeline for Hi-C + microscopy analysis.

> *This is an independent demo project and is not affiliated with, endorsed by, or reviewed by IMBA or the Gerlich lab.*

## Overview

This repository contains two integrated pipelines:

1. **Hi-C Genomics Pipeline** — Load, QC, and visualize Hi-C contact matrices using [cooltools](https://cooltools.readthedocs.io/) and the [cooler](https://github.com/open2c/cooler) format (commonly used in Hi-C analysis; conceptually compatible with interactive exploration tools such as [HiCognition](https://www.hicognition.com)).
2. **Microscopy Pipeline** — Segment nuclei from fluorescence microscopy images using [StarDist](https://github.com/stardist/stardist), with feature extraction and visualization.

Both pipelines feed into a unified **Streamlit dashboard** for interactive exploration.

### Dashboard

The dashboard (`streamlit run app.py`) has five sections (sidebar navigation):

- **QC Dashboard** — Shared QC view for both pipelines; pass/warn/fail flags and thresholds
- **Hi-C Explorer** — Interactive contact heatmap, distance-decay plot, and QC stats table
- **Nuclei Analyzer** — Nuclei features table, per-image summary stats, and segmentation overlays
- **AI Integration** — Upload microscopy images for AI phenotype analysis (OpenAI Vision); structured outputs (nuclei count, morphology, phenotype); rate limiting for public deployment
- **About** — Joint workflow diagram, setup instructions, and privacy/terms

## Sample Outputs

**Hi-C contact matrix** (HFF Micro-C, chr2/chr17):

![Hi-C heatmap](docs/sample_outputs/heatmap.png)

**Nuclei segmentation overlays** (BBBC039, U2OS Hoechst):

![Nuclei overlay 1](docs/sample_outputs/nuclei_overlay_1.png)

![Nuclei overlay 2](docs/sample_outputs/nuclei_overlay_2.png)

## Setup

```bash
cd chromosome-cell-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: Conda (for cooltools compatibility)

```bash
conda env create -f environment.yml
conda activate chrom-cell
```

## Usage

### Quick start (all steps)

```bash
python scripts/04_generate_outputs.py   # Download + run both pipelines
streamlit run app.py
```

### Step by step

1. **Download data**
   ```bash
   python scripts/01_download_data.py
   ```
   Downloads HFF Micro-C (~145 MB) via cooltools and BBBC039 (~78 MB) from Broad Institute. Data is fetched at runtime; not included in this repo. See [Data Sources](#data-sources) for citations and terms.

2. **Run pipelines**
   ```bash
   python scripts/02_hic_pipeline.py
   python scripts/03_microscopy_pipeline.py
   ```

3. **VLM analysis (optional)** — requires `OPENAI_API_KEY`; set `ANTHROPIC_API_KEY` if using Anthropic backend
   ```bash
   export OPENAI_API_KEY=your_key
   python scripts/05_vlm_analysis.py
   ```

4. **Launch dashboard**
   ```bash
   streamlit run app.py
   ```

## Generated Outputs

| Pipeline | Outputs |
|----------|---------|
| **Hi-C** | `output/hic/heatmap.png`, `heatmap.html`, `distance_decay.html`, `qc_stats.csv` |
| **Microscopy** | `output/microscopy/nuclei_features.csv`, `summary_stats.csv`, `overlays/*.png` |
| **VLM** | `output/microscopy/vlm_output.csv` (descriptions, embeddings, nuclei_count, morphology, phenotype) |
| **Integration** | In-dashboard only (morphology stratification + synthetic Hi-C + VLM phenotypes) |

## Configuration

Key options in `config.yaml`:

- **Hi-C:** `resolution` (100 kb), `chromosomes` (chr2, chr17), `download_dataset`
- **Microscopy:** `subset_size` (20 images for demo; set to `null` for all), `stardist_model`
- **VLM:** `backend` (openai/anthropic), `model`, `api_key_env`, `max_images`, `prompt`, `structured_output` (JSON schema), `server_key_only` (public deploy)
- **Rate limiting:** `rate_limit.max_analyses_per_session`, `min_seconds_between_analyses`
- **QC thresholds:** `qc_thresholds.hic` (min_bins, max_sparsity), `qc_thresholds.microscopy` (min_nuclei_per_image, min_images, circularity range)

If StarDist or TensorFlow is unavailable, the microscopy pipeline falls back to watershed-based segmentation.

## Limitations

Demo pipeline; not optimized for production-scale datasets yet.

## Reproducibility

- **Tested:** Python 3.10+, macOS / Linux
- **Dependencies:** See `requirements.txt` (minimum versions; pin exact versions for strict reproducibility)
- **Expected runtime:** ~5–10 min for full pipeline (download + both analyses)
- **Memory:** ~2–4 GB peak

## What This Demonstrates

- **End-to-end pipelines** — Raw data → QC → analysis → visualization
- **Robust Python tooling** — Config-driven, documented, reproducible
- **Microscopy + genomics** — Bridges chromosome organization and quantitative image analysis
- **Standard Hi-C tooling** — Uses cooler format and cooltools workflows; designed for conceptual compatibility with interactive exploration tools such as HiCognition

## Data Sources

Datasets are **downloaded at runtime**; not redistributed in this repository.

| Dataset | Source | License / Terms |
|---------|--------|-----------------|
| **HFF Micro-C** | [cooltools](https://cooltools.readthedocs.io/) test data; 4D Nucleome / open2c ecosystem | 4DN data use policy |
| **BBBC039** | [Broad Institute](https://bbbc.broadinstitute.org/BBBC039) — U2OS nuclei, Hoechst stain | CC0 1.0 (public domain) |

**Citations:** BBBC — Ljosa V, Sokolnicki KL, Carpenter AE (2012). *Nature Methods* 9(7):637. HFF Micro-C — 4D Nucleome Consortium; see cooltools documentation.

## Deploying Publicly

See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying to Streamlit Community Cloud or other hosts. [PUBLIC_DEPLOYMENT_PLAN.md](PUBLIC_DEPLOYMENT_PLAN.md) covers best practices (rate limiting, secrets, cost control). [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md) is a quick pre-deploy checklist. The app includes `deploy_data/` with sample outputs so it works out of the box for public visitors.

## Project Structure

```
chromosome-cell-pipeline/
├── app.py              # Streamlit dashboard
├── config.yaml         # Paths and parameters
├── requirements.txt
├── environment.yml     # Optional conda env
├── CITATION.cff        # Citation metadata
├── deploy_data/        # Sample outputs for public deployment
├── docs/
│   └── sample_outputs/ # Example figures for README
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_hic_pipeline.py
│   ├── 03_microscopy_pipeline.py
│   ├── 04_generate_outputs.py
│   └── 05_vlm_analysis.py
├── data/               # Downloaded (gitignored)
└── output/             # Generated figures (gitignored)
```

## Author

**Almotasem Bellah Younis** — [ORCID](https://orcid.org/0000-0003-2070-2811) · [GitHub](https://github.com/mojo8787)

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.

## Releasing / Zenodo

To create a new version for Zenodo:

1. Update `CHANGELOG.md` with the release date
2. Commit all changes and push to `main`
3. Create a GitHub release: **Releases** → **Create a new release** → tag `v1.1.0` (or next version)
4. If Zenodo is connected to your repo, it will automatically archive the new version and assign a DOI

See [Zenodo-GitHub integration](https://docs.zenodo.org/guides/github/) for setup.

## License

MIT
