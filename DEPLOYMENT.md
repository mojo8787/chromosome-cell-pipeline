# Deploying the Chromosome + Cell Pipeline (Public)

This guide explains how to deploy the Streamlit dashboard publicly so others can use it without running anything locally.

> **📋 Full checklist:** See [PUBLIC_DEPLOYMENT_PLAN.md](PUBLIC_DEPLOYMENT_PLAN.md) for best practices, security, and cost control.

## Option 1: Streamlit Community Cloud (Recommended)

[Streamlit Community Cloud](https://share.streamlit.io/) offers free hosting for Streamlit apps. **Live demo:** [chromosome-cell-pipeline.streamlit.app](https://chromosome-cell-pipeline-potip23aylajefym9mqh67.streamlit.app)

### Prerequisites

- GitHub account
- Repository pushed to GitHub (e.g. `mojo8787/chromosome-cell-pipeline`)

### Steps

1. **Go to [share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.

2. **Click "New app"** and connect your repository:
   - **Repository:** `your-username/chromosome-cell-pipeline`
   - **Branch:** `main` (or your default branch)
   - **Main file path:** `app.py`
   - **App URL:** choose a subdomain (e.g. `chromosome-cell-pipeline`)

3. **Click "Deploy"**. The first build may take 5–10 minutes due to dependencies (cooltools, StarDist, etc.).

4. **Optional – Advanced settings:**
   - **Python version:** 3.10 or 3.11 (recommended)
   - **Secrets (recommended for public use):** Add your OpenAI API key so visitors can run VLM without entering a key:
     - Click the app → **Settings** (⚙️) → **Secrets**
     - Add:
       ```toml
       OPENAI_API_KEY = "sk-your-key-here"
       ```
     - Save. With this set, the app uses the server-side key automatically; all visitors can run VLM without entering a key (you pay for usage).

### After Deployment

- The app will be available at `https://your-app-name.streamlit.app`.
- **Hi-C and Microscopy tabs:** Show sample data if `deploy_data/` is present; otherwise they show "Run the pipeline first".
- **VLM (Integration tab):**
  - **Option A (with secrets):** If you add `OPENAI_API_KEY` to Streamlit secrets, visitors can use VLM without entering a key.
  - **Option B (without secrets):** Visitors enter their own API key in the password field.
  - **Image upload:** Visitors can upload their own microscopy images (PNG, JPG, TIFF) for VLM phenotype analysis.

### Build Notes

- Dependencies (TensorFlow, cooltools, etc.) can make the build slow. If the build fails, try:
  - Pinning versions in `requirements.txt`
  - Using Python 3.10
- The app uses `deploy_data/` as fallback when `output/` is missing, so the public demo works without running the pipeline.

---

## Option 2: Hugging Face Spaces

1. Create a new [Space](https://huggingface.co/spaces) and choose **Streamlit**.
2. Clone your repo or upload the project files.
3. Set the Space to use `app.py` as the main script.
4. Add `requirements.txt` in the Space root.

---

## Option 3: Self-Hosted (Docker, Cloud Run, etc.)

For full control, you can run the app in a container:

```bash
# Example: run with Docker (create a Dockerfile that installs deps and runs streamlit)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Security Notes for Public Deployment

- **API keys:** Never commit API keys. The app asks users to enter their OpenAI key in the UI; it is used only in memory and not stored.
- **Data:** The app does not collect or store user data. VLM descriptions and embeddings are generated per session.
