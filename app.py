"""
Chromosome + Cell Mini-Pipeline — Streamlit Dashboard.

Unified interface for Hi-C chromatin exploration and nuclei segmentation analysis.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _resolve_output(subdir: str) -> Path:
    """Use output/ if it has data, else deploy_data/ for public demo."""
    out = ROOT / "output" / subdir
    deploy = ROOT / "deploy_data" / subdir
    if subdir == "hic":
        if (out / "qc_stats.csv").exists():
            return out
        if (deploy / "qc_stats.csv").exists():
            return deploy
    elif subdir == "microscopy":
        if (out / "nuclei_features.csv").exists():
            return out
        if (deploy / "nuclei_features.csv").exists():
            return deploy
    return out


import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Chromosome + Cell Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more polished, encouraging UI
st.markdown(
    """
<style>
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #0d7377 0%, #14a3a8 50%, #0d7377 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(13, 115, 119, 0.25);
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1rem;
        margin: 0.5rem 0 0;
        opacity: 0.95;
    }
    /* Card-style sections */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafa;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0d7377 !important;
    }
    /* Success / info boxes */
    .stSuccess, .stInfo {
        border-radius: 10px;
        border-left: 4px solid #0d7377;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(13, 115, 119, 0.3);
    }
    /* Section headers */
    h2, h3 {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }
    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #0d7377;
        border-radius: 12px;
        padding: 2rem;
        background: #f8fafa;
    }
    /* CTA callout */
    .cta-box {
        background: linear-gradient(135deg, #e8f6f6 0%, #f0f7f7 100%);
        border: 1px solid #0d7377;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar — clickable navigation
with st.sidebar:
    st.markdown("## 🧬 Chromosome + Cell Pipeline")
    st.markdown("---")
    section = st.radio(
        "**Go to**",
        [
            "📊 QC Dashboard",
            "🔬 Hi-C Explorer",
            "🔭 Nuclei Analyzer",
            "✨ AI Integration",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
        key="sidebar_nav",
    )

# Hero section
st.markdown(
    """
<div class="hero">
    <h1>🧬 Chromosome + Cell Pipeline</h1>
    <p>Bridge 3D imaging with genomic insights — Hi-C chromatin analysis, nuclei segmentation, and AI-powered phenotype interpretation</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- QC Dashboard ---
if section == "📊 QC Dashboard":
    st.header("QC Dashboard")
    import yaml

    config_path = ROOT / "config.yaml"
    thresholds = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        thresholds = cfg.get("qc_thresholds", {})

    hic_output = _resolve_output("hic")
    micro_output = _resolve_output("microscopy")
    hic_ok = (hic_output / "qc_stats.csv").exists()
    micro_ok = (micro_output / "nuclei_features.csv").exists()

    if not hic_ok and not micro_ok:
        st.info(
            "**Welcome!** Run the pipeline to generate data: `python scripts/04_generate_outputs.py` "
            "or run the download and analysis scripts step by step."
        )
    else:
        col_hic, col_micro = st.columns(2)
        flags = []

        with col_hic:
            st.subheader("Hi-C QC")
            if hic_ok:
                import pandas as pd

                qc = pd.read_csv(hic_output / "qc_stats.csv")
                row = qc.iloc[0]
                bins = int(row.get("n_bins", 0))
                sparsity = float(row.get("sparsity", 0))
                th = thresholds.get("hic", {})
                min_bins = th.get("min_bins", 100)
                max_sparsity = th.get("max_sparsity", 0.9999)

                c1, c2 = st.columns(2)
                c1.metric("Bins", f"{bins:,}")
                c2.metric("Sparsity", f"{sparsity:.4f}")

                if bins >= min_bins and sparsity <= max_sparsity:
                    st.success("Pass")
                elif bins < min_bins:
                    st.error("Fail")
                    flags.append(f"Hi-C: bins {bins} below min {min_bins}")
                else:
                    st.warning("Warn")
                    flags.append(
                        f"Hi-C: sparsity {sparsity:.4f} exceeds {max_sparsity}"
                    )
            else:
                st.warning("Hi-C pipeline not run")
                flags.append("Hi-C: pipeline not run")

        with col_micro:
            st.subheader("Microscopy QC")
            if micro_ok:
                import pandas as pd

                df = pd.read_csv(micro_output / "nuclei_features.csv")
                n_images = df["image"].nunique()
                counts = df.groupby("image").size()
                mean_nuclei = counts.mean()
                circ_mean = df["circularity"].mean()
                circ_min = df["circularity"].min()
                circ_max = df["circularity"].max()

                th = thresholds.get("microscopy", {})
                min_nuclei = th.get("min_nuclei_per_image", 20)
                min_imgs = th.get("min_images", 5)
                circ_lo = th.get("circularity_low", 0.6)
                circ_hi = th.get("circularity_high", 0.95)

                c1, c2, c3 = st.columns(3)
                c1.metric("Images", n_images)
                c2.metric("Mean nuclei/image", f"{mean_nuclei:.0f}")
                c3.metric("Circularity", f"{circ_mean:.2f}")

                micro_pass = True
                if n_images < min_imgs:
                    micro_pass = False
                    flags.append(f"Microscopy: {n_images} images below min {min_imgs}")
                low_n = counts[counts < min_nuclei]
                if len(low_n) > 0:
                    micro_pass = False
                    for img in low_n.index[:3]:
                        flags.append(
                            f"Image {str(img)[:35]}...: {int(low_n[img])} nuclei "
                            f"(below {min_nuclei})"
                        )
                if circ_min < circ_lo or circ_max > circ_hi:
                    micro_pass = False
                    flags.append(
                        f"Circularity range [{circ_min:.2f}, {circ_max:.2f}] "
                        f"outside [{circ_lo}, {circ_hi}]"
                    )

                if micro_pass:
                    st.success("Pass")
                else:
                    st.error("Fail")
            else:
                st.warning("Microscopy pipeline not run")
                flags.append("Microscopy: pipeline not run")

        if flags:
            st.subheader("Flags")
            for f in flags:
                st.warning(f)

# --- Hi-C Explorer ---
elif section == "🔬 Hi-C Explorer":
    st.header("Hi-C Chromatin Explorer")
    hic_output = _resolve_output("hic")
    has_heatmap = (hic_output / "heatmap.html").exists() or (
        hic_output / "heatmap.png"
    ).exists()
    if has_heatmap:
        import pandas as pd

        qc = (
            pd.read_csv(hic_output / "qc_stats.csv")
            if (hic_output / "qc_stats.csv").exists()
            else pd.DataFrame()
        )
        if not qc.empty:
            st.subheader("QC summary")
            row = qc.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Resolution", f"{int(row.get('binsize', 0)) / 1000:.0f} kb")
            c2.metric("Bins", f"{int(row.get('n_bins', 0)):,}")
            c3.metric("Pixels", f"{int(row.get('n_pixels', 0)):,}")
            c4.metric("Sparsity", f"{float(row.get('sparsity', 0)):.4f}")
        if (hic_output / "heatmap.html").exists():
            with open(hic_output / "heatmap.html") as f:
                st.components.v1.html(f.read(), height=550, scrolling=False)
        else:
            st.image(str(hic_output / "heatmap.png"), width="stretch")
        if (hic_output / "distance_decay.html").exists():
            with open(hic_output / "distance_decay.html") as f:
                st.components.v1.html(f.read(), height=450, scrolling=False)
        if not qc.empty:
            st.subheader("QC stats (full)")
            st.dataframe(qc, width="stretch", hide_index=True)
    else:
        st.info(
            "Run the Hi-C pipeline to see the chromatin heatmap: `python scripts/01_download_data.py` "
            "then `python scripts/02_hic_pipeline.py`"
        )

# --- Nuclei Analyzer ---
elif section == "🔭 Nuclei Analyzer":
    st.header("Nuclei Segmentation Analyzer")
    micro_output = _resolve_output("microscopy")
    if (micro_output / "nuclei_features.csv").exists():
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv(micro_output / "nuclei_features.csv")

        # Summary metrics cards
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total nuclei", f"{len(df):,}")
        with c2:
            st.metric("Images", df["image"].nunique())
        with c3:
            st.metric("Mean area (px²)", f"{df['area'].mean():.0f}")
        with c4:
            st.metric("Mean circularity", f"{df['circularity'].mean():.2f}")

        # Feature distribution plots
        st.subheader("Feature distributions")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_area = px.histogram(
                df,
                x="area",
                nbins=40,
                title="Area (px²)",
                labels={"area": "Area"},
                color_discrete_sequence=["#2ecc71"],
            )
            fig_area.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
            st.plotly_chart(fig_area, width="stretch")
        with col_b:
            fig_circ = px.histogram(
                df,
                x="circularity",
                nbins=40,
                title="Circularity",
                labels={"circularity": "Circularity"},
                color_discrete_sequence=["#3498db"],
            )
            fig_circ.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
            st.plotly_chart(fig_circ, width="stretch")

        # Nuclei count per image (bar chart)
        counts = df.groupby("image").size().reset_index(name="n_nuclei")
        fig_counts = px.bar(
            counts,
            x="image",
            y="n_nuclei",
            title="Nuclei per image",
            labels={"n_nuclei": "Count", "image": "Image"},
            color_discrete_sequence=["#9b59b6"],
        )
        fig_counts.update_layout(
            xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=80), height=300
        )
        st.plotly_chart(fig_counts, width="stretch")

        st.subheader("Nuclei features (table)")
        st.dataframe(df.head(100), width="stretch", hide_index=True)
        if (micro_output / "summary_stats.csv").exists():
            summary = pd.read_csv(micro_output / "summary_stats.csv", index_col=0)
            st.subheader("Summary per image")
            st.dataframe(summary, width="stretch", hide_index=True)
        overlays = list((micro_output / "overlays").glob("*.png"))
        if overlays:
            st.subheader("Segmentation overlays (raw → segmented)")
            sel = st.selectbox(
                "Select image",
                options=overlays,
                format_func=lambda p: p.stem[:60] + ("..." if len(p.stem) > 60 else ""),
                key="overlay_sel",
            )
            if sel:
                st.image(str(sel), width="stretch")
            cols = st.columns(min(3, len(overlays)))
            for i, p in enumerate(overlays[:6]):
                cols[i % 3].image(str(p), caption=p.stem[:30] + "...", width="stretch")
    else:
        st.info(
            "Run the microscopy pipeline to see nuclei segmentation: `python scripts/01_download_data.py` "
            "then `python scripts/03_microscopy_pipeline.py`"
        )

# --- Integration ---
elif section == "✨ AI Integration":
    st.header("AI-Powered Phenotype Analysis")
    st.markdown(
        "Use Vision-Language Models to automatically describe nuclei morphology, chromatin distribution, "
        "and phenotypic features in your microscopy images — no manual annotation needed."
    )
    micro_output = _resolve_output("microscopy")

    # Rate limiting: load config and init session state
    import time
    import yaml as _yaml

    _rate_cfg = {}
    if (ROOT / "config.yaml").exists():
        with open(ROOT / "config.yaml") as _f:
            _rate_cfg = (_yaml.safe_load(_f) or {}).get("rate_limit", {})
    _max_analyses = _rate_cfg.get("max_analyses_per_session", 10)
    _min_seconds = _rate_cfg.get("min_seconds_between_analyses", 30)
    _warn_at = _rate_cfg.get("warning_at_remaining", 2)
    if "vlm_analysis_count" not in st.session_state:
        st.session_state["vlm_analysis_count"] = 0
    if "vlm_last_analysis_time" not in st.session_state:
        st.session_state["vlm_last_analysis_time"] = 0.0

    def _check_rate_limit():
        count = st.session_state["vlm_analysis_count"]
        last = st.session_state["vlm_last_analysis_time"]
        elapsed = time.time() - last
        if count >= _max_analyses:
            return (
                False,
                f"Session limit reached ({_max_analyses} analyses). Refresh the page to reset.",
            )
        if last > 0 and elapsed < _min_seconds:
            wait = int(_min_seconds - elapsed)
            return False, f"Please wait {wait} seconds before another analysis."
        return True, None

    def _record_analysis():
        st.session_state["vlm_analysis_count"] = (
            st.session_state.get("vlm_analysis_count", 0) + 1
        )
        st.session_state["vlm_last_analysis_time"] = time.time()

    # VLM output written to output/microscopy/; may also exist in deploy_data
    vlm_output_path = ROOT / "output" / "microscopy" / "vlm_output.csv"
    if not vlm_output_path.exists():
        vlm_output_path = micro_output / "vlm_output.csv"
    overlay_dir = micro_output / "overlays"

    # Run VLM Analysis from Streamlit
    st.markdown("---")
    st.subheader("Get started")
    # Use Streamlit secrets (server-side key); server_key_only=true hides user input
    try:
        api_key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key_from_secrets = ""
    vlm_config_yaml = {}
    if (ROOT / "config.yaml").exists():
        import yaml as _yaml

        with open(ROOT / "config.yaml") as _f:
            vlm_config_yaml = _yaml.safe_load(_f) or {}
    server_key_only = vlm_config_yaml.get("vlm", {}).get("server_key_only", False)
    api_key = api_key_from_secrets
    if not api_key and not server_key_only:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key to enable AI analysis. Your key is never stored.",
            key="vlm_api_key",
        )
    if api_key:
        st.success("✓ API key configured — you're ready to analyze images.")
    elif server_key_only:
        st.error(
            "AI analysis is not configured. The administrator must add OPENAI_API_KEY to Streamlit secrets."
        )

    # Usage / rate limit warning
    _remaining = _max_analyses - st.session_state["vlm_analysis_count"]
    if _remaining <= _warn_at and _remaining > 0:
        st.warning(
            f"⚠️ {_remaining} analysis(es) remaining this session. Limit: {_max_analyses} per session."
        )
    elif _remaining <= 0:
        st.error(
            f"Session limit reached ({_max_analyses} analyses). Refresh the page to get a new allowance."
        )
    else:
        st.caption(
            f"Usage: {st.session_state['vlm_analysis_count']}/{_max_analyses} analyses this session • min {_min_seconds}s between runs"
        )

    # Option 1: Upload your own images
    st.markdown("#### 📤 Upload your microscopy images")
    st.markdown(
        "Drop your images below for instant AI phenotype analysis. Supports PNG, JPG, and TIFF."
    )
    uploaded_files = st.file_uploader(
        "Choose microscopy images (PNG, JPG, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="vlm_upload",
    )
    if st.button("🔍 Analyze with AI", key="vlm_upload_btn", type="primary"):
        ok, err = _check_rate_limit()
        if not ok:
            st.error(err)
        elif not api_key or not str(api_key).strip():
            if server_key_only:
                st.error("AI analysis is not configured. Contact the administrator.")
            else:
                st.warning(
                    "Please enter your OpenAI API key or configure Streamlit secrets."
                )
        elif not uploaded_files:
            st.warning("Please upload at least one image.")
        else:
            import tempfile
            import importlib.util
            import yaml

            with st.spinner("Analyzing uploaded images..."):
                try:
                    spec = importlib.util.spec_from_file_location(
                        "vlm_script", ROOT / "scripts" / "05_vlm_analysis.py"
                    )
                    vlm_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(vlm_module)
                    with open(ROOT / "config.yaml") as f:
                        vlm_config = yaml.safe_load(f)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir_path = Path(tmpdir).resolve()
                        paths = []
                        img_bytes = {}
                        max_file_size = 50 * 1024 * 1024  # 50 MB per file
                        for i, f in enumerate(uploaded_files[:10]):  # limit to 10
                            data = f.getvalue()
                            if len(data) > max_file_size:
                                continue  # Skip oversized files
                            # Sanitize filename to prevent path traversal
                            safe_name = Path(f.name).name or f"image_{i}"
                            p = (tmpdir_path / safe_name).resolve()
                            if not str(p).startswith(str(tmpdir_path)):
                                continue  # Skip invalid filenames
                            p.write_bytes(data)
                            paths.append(p)
                            # Convert TIFF to PNG for display (Streamlit/PIL fails on raw TIFF)
                            ext = Path(safe_name).suffix.lower()
                            if ext in (".tif", ".tiff"):
                                try:
                                    import io
                                    import numpy as np
                                    from PIL import Image

                                    try:
                                        import tifffile

                                        arr = tifffile.imread(io.BytesIO(data))
                                    except Exception:
                                        arr = np.array(Image.open(io.BytesIO(data)))
                                    arr = np.atleast_2d(arr).astype(np.float64)
                                    if arr.ndim > 2:
                                        arr = arr.squeeze()
                                        if arr.ndim > 2:
                                            arr = arr[0]
                                    p2, p98 = np.percentile(arr, (2, 98))
                                    if p98 > p2:
                                        arr = np.clip(
                                            (arr - p2) / (p98 - p2) * 255, 0, 255
                                        )
                                    arr = arr.astype(np.uint8)
                                    img = Image.fromarray(arr, mode="L").convert("RGB")
                                    buf = io.BytesIO()
                                    img.save(buf, format="PNG")
                                    img_bytes[safe_name] = buf.getvalue()
                                except Exception:
                                    img_bytes[safe_name] = data
                            else:
                                img_bytes[safe_name] = data
                        df = None
                        if paths:
                            df = vlm_module.analyze_images(
                                paths, str(api_key).strip(), vlm_config
                            )
                            st.session_state["vlm_uploaded_images"] = img_bytes
                    if df is not None and len(df) > 0:
                        _record_analysis()
                        st.session_state["vlm_uploaded_results"] = df
                        st.success(
                            f"✓ Successfully analyzed {len(df)} image(s)! Results below."
                        )
                        st.rerun()
                    elif not paths:
                        st.error("No valid images to analyze.")
                    else:
                        st.error("Analysis failed. Check the terminal for details.")
                except Exception as e:
                    st.error(f"Analysis failed: **{str(e)}**")

    # Option 2: Run on pipeline overlays (when available)
    if overlay_dir.exists() and list(overlay_dir.glob("*.png")):
        st.markdown("---")
        st.markdown("#### 📁 Or analyze pipeline overlays")
        st.caption("Use pre-segmented overlays from the microscopy pipeline.")
        if st.button("Run AI on pipeline overlays", key="vlm_run_btn"):
            ok, err = _check_rate_limit()
            if not ok:
                st.error(err)
            elif api_key and str(api_key).strip():
                import importlib.util
                import yaml

                with st.spinner("Running VLM analysis (this may take a minute)..."):
                    spec = importlib.util.spec_from_file_location(
                        "vlm_script", ROOT / "scripts" / "05_vlm_analysis.py"
                    )
                    vlm_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(vlm_module)
                    with open(ROOT / "config.yaml") as f:
                        vlm_config = yaml.safe_load(f)
                    # Temporarily set env for the pipeline
                    import os

                    prev_key = os.environ.get("OPENAI_API_KEY")
                    os.environ["OPENAI_API_KEY"] = str(api_key).strip()
                    try:
                        ov_dir = micro_output / "overlays"
                        result = vlm_module.run_pipeline(
                            vlm_config,
                            api_key_override=str(api_key).strip(),
                            overlay_dir_override=ov_dir if ov_dir.exists() else None,
                        )
                    finally:
                        if prev_key is not None:
                            os.environ["OPENAI_API_KEY"] = prev_key
                        else:
                            os.environ.pop("OPENAI_API_KEY", None)
                if result is not None:
                    _record_analysis()
                    st.success("VLM analysis complete. Refreshing...")
                    st.rerun()
                else:
                    st.error("VLM analysis failed. Check the terminal for details.")
            else:
                st.warning("Please enter your OpenAI API key.")

    # Generative phenotyping: Hi-C + microscopy (joint VLM analysis)
    hic_output = _resolve_output("hic")
    hic_heatmap_path = hic_output / "heatmap.png"
    if not hic_heatmap_path.exists():
        hic_heatmap_path = ROOT / "deploy_data" / "hic" / "heatmap.png"
    if hic_heatmap_path.exists() and api_key:
        st.markdown("---")
        with st.expander("🔬 Generative phenotyping: Hi-C + microscopy", expanded=False):
            st.caption(
                "Send both a Hi-C contact map and a microscopy image to the VLM. "
                "Ask about relationships between chromatin organization and nuclear morphology."
            )
            joint_micro_upload = st.file_uploader(
                "Microscopy image for joint analysis",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="joint_micro_upload",
            )
            overlay_paths = sorted(overlay_dir.glob("*.png")) if overlay_dir.exists() else []
            overlay_choice = None
            if overlay_paths and not joint_micro_upload:
                st.caption("Or select from pipeline overlays:")
                overlay_choice = st.selectbox(
                    "Overlay",
                    [None] + overlay_paths,
                    format_func=lambda p: "—" if p is None else p.name,
                    key="joint_overlay",
                )
            if st.button("Run generative phenotyping", key="joint_btn"):
                ok, err = _check_rate_limit()
                if not ok:
                    st.error(err)
                elif not joint_micro_upload and not overlay_choice:
                    st.warning("Upload a microscopy image or select an overlay.")
                else:
                    import tempfile
                    import importlib.util
                    import yaml

                    micro_path = None
                    if joint_micro_upload:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            micro_path = Path(tmpdir) / Path(joint_micro_upload.name).name
                            micro_path.write_bytes(joint_micro_upload.getvalue())
                            with st.spinner("Analyzing Hi-C + microscopy (generative phenotyping)..."):
                                try:
                                    spec = importlib.util.spec_from_file_location(
                                        "vlm_script", ROOT / "scripts" / "05_vlm_analysis.py"
                                    )
                                    vlm_module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(vlm_module)
                                    with open(ROOT / "config.yaml") as f:
                                        vlm_config = yaml.safe_load(f)
                                    prompt = vlm_config.get("vlm", {}).get(
                                        "joint_prompt",
                                        "Describe the relationship between the chromatin contact patterns in the Hi-C map and the nuclear staining in the microscopy image. Be concise.",
                                    )
                                    model = vlm_config.get("vlm", {}).get("model", "gpt-4o")
                                    result = vlm_module.get_joint_vlm_description_openai(
                                        micro_path,
                                        hic_heatmap_path,
                                        prompt,
                                        model,
                                        str(api_key).strip(),
                                    )
                                    _record_analysis()
                                    st.session_state["joint_phenotyping_result"] = result
                                    st.success("✓ Generative phenotyping complete.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Analysis failed: **{str(e)}**")
                    elif overlay_choice and overlay_choice.exists():
                        with st.spinner("Analyzing Hi-C + microscopy (generative phenotyping)..."):
                            try:
                                spec = importlib.util.spec_from_file_location(
                                    "vlm_script", ROOT / "scripts" / "05_vlm_analysis.py"
                                )
                                vlm_module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(vlm_module)
                                with open(ROOT / "config.yaml") as f:
                                    vlm_config = yaml.safe_load(f)
                                prompt = vlm_config.get("vlm", {}).get(
                                    "joint_prompt",
                                    "Describe the relationship between the chromatin contact patterns in the Hi-C map and the nuclear staining in the microscopy image. Be concise.",
                                )
                                model = vlm_config.get("vlm", {}).get("model", "gpt-4o")
                                result = vlm_module.get_joint_vlm_description_openai(
                                    overlay_choice,
                                    hic_heatmap_path,
                                    prompt,
                                    model,
                                    str(api_key).strip(),
                                )
                                _record_analysis()
                                st.session_state["joint_phenotyping_result"] = result
                                st.success("✓ Generative phenotyping complete.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Analysis failed: **{str(e)}**")
            if "joint_phenotyping_result" in st.session_state:
                st.markdown("**VLM response:**")
                st.write(st.session_state["joint_phenotyping_result"])

    # VLM Phenotype Descriptions (when available: from upload or pipeline output)
    import pandas as pd

    vlm_df = None
    if "vlm_uploaded_results" in st.session_state:
        vlm_df = st.session_state["vlm_uploaded_results"]
    elif vlm_output_path.exists():
        vlm_df = pd.read_csv(vlm_output_path)

    if vlm_df is not None and len(vlm_df) > 0:
        import json

        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from sklearn.cluster import KMeans

        st.subheader("AI phenotype descriptions")
        st.caption(
            "Each image was analyzed by a Vision-Language Model. View image and description side by side."
        )
        if "vlm_uploaded_results" in st.session_state:
            if st.button("Clear uploaded results", key="vlm_clear"):
                del st.session_state["vlm_uploaded_results"]
                st.session_state.pop("vlm_uploaded_images", None)
                st.rerun()
        # View mode: cards or one-at-a-time
        uploaded_imgs = st.session_state.get("vlm_uploaded_images", {})
        view_mode = st.radio(
            "View", ["All cards", "One at a time"], horizontal=True, key="vlm_view_mode"
        )
        rows_list = list(vlm_df.iterrows())
        if view_mode == "One at a time" and len(rows_list) > 1:
            sel_idx = st.selectbox(
                "Select image",
                range(len(rows_list)),
                format_func=lambda i: rows_list[i][1]["image"],
                key="vlm_sel",
            )
            rows_list = [rows_list[sel_idx]]
        for idx, (_, row) in enumerate(rows_list):
            img_name = row["image"]
            desc = row["description"]
            # Get image: uploaded bytes or overlay file
            img_src = None
            if img_name in uploaded_imgs:
                img_src = uploaded_imgs[img_name]
            else:
                ov_path = overlay_dir / img_name
                if ov_path.exists():
                    img_src = ov_path.read_bytes()
            with st.container(border=True):
                col_img, col_txt = st.columns([1, 2], gap="large")
                with col_img:
                    if img_src:
                        try:
                            st.image(
                                img_src,
                                caption=img_name[:40]
                                + ("..." if len(img_name) > 40 else ""),
                                width="stretch",
                                output_format="PNG",
                            )
                        except Exception:
                            st.caption(f"📷 {img_name} (preview unavailable)")
                    else:
                        st.caption(f"📷 {img_name}")
                with col_txt:
                    st.markdown(f"**{img_name}**")
                    has_structured = "nuclei_count" in row and "morphology" in row
                    if has_structured:
                        n_count = row.get("nuclei_count", 0)
                        try:
                            n_count = int(float(n_count))
                        except (ValueError, TypeError):
                            n_count = 0
                        st.metric("Nuclei count", n_count)
                        st.markdown("**Morphology:**")
                        st.write(row.get("morphology", ""))
                        st.markdown("**Phenotype:**")
                        st.write(row.get("phenotype", ""))
                    else:
                        st.write(desc)
        # VLM-derived strata (k-means on embeddings)
        embeddings = vlm_df["embedding"].apply(json.loads).tolist()
        if len(embeddings) >= 2:
            n_clusters = min(3, len(embeddings))
            X = np.array(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            vlm_df = vlm_df.copy()
            vlm_df["vlm_cluster"] = kmeans.fit_predict(X)
            cluster_labels = {i: f"Cluster {i + 1}" for i in range(n_clusters)}
            vlm_df["vlm_stratum"] = vlm_df["vlm_cluster"].map(cluster_labels)
            vlm_counts = vlm_df["vlm_stratum"].value_counts().sort_index()
            st.subheader("VLM-derived strata (embedding clusters)")
            colors = ["#2ecc71", "#3498db", "#e74c3c"][:n_clusters]
            fig_vlm = go.Figure(
                data=[
                    go.Bar(
                        x=list(vlm_counts.index),
                        y=vlm_counts.values,
                        marker_color=colors,
                    )
                ]
            )
            fig_vlm.update_layout(
                xaxis_title="Cluster",
                yaxis_title="Images",
                height=250,
            )
            st.plotly_chart(fig_vlm, width="stretch")
    else:
        st.info(
            "👆 **Upload images above** and click **Analyze with AI** to get started. "
            "Or run the pipeline overlays analysis if you have pre-segmented data."
        )

    if (micro_output / "nuclei_features.csv").exists():
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        df = pd.read_csv(micro_output / "nuclei_features.csv")

        # Stratify by area and circularity (terciles)
        area_33 = df["area"].quantile(1 / 3)
        area_67 = df["area"].quantile(2 / 3)
        circ_33 = df["circularity"].quantile(1 / 3)
        circ_67 = df["circularity"].quantile(2 / 3)

        def stratum(row):
            if row["area"] <= area_33 and row["circularity"] >= circ_67:
                return "Compact"
            if row["area"] >= area_67 and row["circularity"] <= circ_33:
                return "Dispersed"
            return "Intermediate"

        df["stratum"] = df.apply(stratum, axis=1)
        strata_counts = df.groupby("stratum").size()

        st.info(
            "Conceptual demonstration. Synthetic Hi-C patterns illustrate how "
            "morphology strata might correlate with chromatin compaction. "
            "Real integration requires matched imaging + Hi-C data."
        )

        # Synthetic Hi-C: power-law decay contacts ~ |i-j|^(-alpha)
        def synthetic_hic(n=128, alpha=1.0):
            mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    d = max(j - i, 1)
                    val = d ** (-alpha)
                    mat[i, j] = mat[j, i] = val
            return np.log1p(mat)

        alphas = {"Compact": 1.2, "Intermediate": 0.9, "Dispersed": 0.6}
        st.subheader("Synthetic Hi-C by morphology stratum")

        cols = st.columns(3)
        for idx, (name, alpha) in enumerate(alphas.items()):
            with cols[idx]:
                n_count = strata_counts.get(name, 0)
                st.caption(f"{name}: {n_count} nuclei (α={alpha})")
                mat = synthetic_hic(128, alpha)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=mat,
                        colorscale="Reds",
                        hoverongaps=False,
                    )
                )
                fig.update_layout(
                    title=f"{name}",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, width="stretch")

        st.subheader("Stratum distribution")
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=list(strata_counts.index),
                    y=strata_counts.values,
                    marker_color=["#2ecc71", "#3498db", "#e74c3c"],
                )
            ]
        )
        fig_bar.update_layout(
            xaxis_title="Stratum",
            yaxis_title="Nuclei count",
            height=300,
        )
        st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info(
            "Run the microscopy pipeline to see morphology–chromatin integration: "
            "`python scripts/03_microscopy_pipeline.py`"
        )

# --- About ---
else:  # ℹ️ About
    st.header("About this pipeline")
    st.markdown(
        "A combined genomics and microscopy analysis platform that bridges "
        "**chromosome organization** (Hi-C) with **quantitative image analysis** (nuclei segmentation) "
        "and **AI-powered phenotype interpretation** (Vision-Language Models)."
    )
    st.info(
        "This is an independent demo project and is not affiliated with, "
        "endorsed by, or reviewed by IMBA or the Gerlich lab."
    )
    with st.expander("📋 Privacy & terms (public deployment)", expanded=False):
        st.markdown("""
        **Data processing:** Images you upload are sent to OpenAI for AI analysis. OpenAI's [privacy policy](https://openai.com/policies/privacy-policy) applies to that processing.

        **Storage:** We do not store your images or descriptions on our servers. Session data is cleared when you leave.

        **Intended use:** For research and educational purposes. This tool is not intended for clinical or medical use. Results are not medical advice.

        **Your responsibility:** Only upload images you have the right to share and analyze.
        """)
    st.subheader("Pipeline workflow")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (x, y, txt) in enumerate(
        [
            (1, 1, "Raw data\n(.mcool, .tif)"),
            (3, 1, "QC\n(Hi-C + microscopy)"),
            (5, 1, "Analysis\n(contacts + segmentation)"),
            (7, 1, "Figures\n(heatmaps, overlays, integration)"),
        ]
    ):
        rect = mpatches.FancyBboxPatch(
            (x - 0.4, y - 0.25),
            0.8,
            0.5,
            boxstyle="round,pad=0.02",
            facecolor="#3498db",
            edgecolor="#2c3e50",
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, color="white")
        if i < 3:
            ax.annotate(
                "",
                xy=(x + 0.5, y),
                xytext=(x + 0.4, y),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2),
            )
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    This demo combines two pipelines:

    1. **Hi-C Genomics** — Load, QC, and visualize Hi-C contact matrices using
       [cooltools](https://cooltools.readthedocs.io/) and the
       [cooler](https://github.com/open2c/cooler) format.

    2. **Microscopy** — Segment nuclei from fluorescence images using
       [StarDist](https://github.com/stardist/stardist), with feature extraction.

    **Data sources:**
    - Hi-C: HFF Micro-C (cooltools test data)
    - Microscopy: BBBC039 — U2OS nuclei, Hoechst (Broad Institute)

    **Setup:**
    ```bash
    pip install -r requirements.txt
    python scripts/01_download_data.py
    python scripts/02_hic_pipeline.py
    python scripts/03_microscopy_pipeline.py
    streamlit run app.py
    ```
    """)
