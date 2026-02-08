# Changelog

All notable changes to this project are documented in this file.

## [1.1.0] — 2025-02-06

### Added

- **AI Integration tab** — Upload microscopy images for AI phenotype analysis (OpenAI Vision)
- **Structured outputs** — JSON schema: `nuclei_count`, `morphology`, `phenotype` (config: `vlm.structured_output`)
- **Sidebar navigation** — Clickable sections instead of top tabs
- **Rate limiting** — Session limits (5 analyses, 60s between runs) to protect API credits
- **Public deployment mode** — `server_key_only: true` hides API key input; uses Streamlit secrets only
- **TIFF support** — 16-bit microscopy TIFFs normalized for display; converted to PNG for API
- **Deployment docs** — `PUBLIC_DEPLOYMENT_PLAN.md`, `DEPLOY_CHECKLIST.md`
- **Privacy & terms** — Expandable section in About tab

### Changed

- **Config** — `rate_limit`, `vlm.server_key_only`, `vlm.structured_output`
- **Image display** — `output_format="PNG"` to avoid PIL/JPEG errors with grayscale images

### Fixed

- OSError when displaying TIFF images in Streamlit (conversion to RGB PNG)
- Ruff format for CI

---

## [1.0.0] — 2024 (initial)

- Hi-C pipeline (cooltools, cooler)
- Microscopy pipeline (StarDist, watershed fallback)
- Streamlit dashboard (QC, Hi-C Explorer, Nuclei Analyzer, Integration, About)
- VLM analysis script (OpenAI/Anthropic)
- Zenodo archive (DOI: 10.5281/zenodo.18500690)
