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
)

st.title("Chromosome + Cell Mini-Pipeline")
st.caption(
    "Hi-C chromatin analysis + microscopy nuclei segmentation | "
    "Uses cooler format; designed for conceptual compatibility with tools such as [HiCognition](https://www.hicognition.com)"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["QC Dashboard", "Hi-C Explorer", "Nuclei Analyzer", "Integration", "About"]
)

# --- QC Dashboard ---
with tab1:
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
            "Run the pipeline first: `python scripts/04_generate_outputs.py` or "
            "`python scripts/01_download_data.py` then `02_hic_pipeline.py` and "
            "`03_microscopy_pipeline.py`"
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
with tab2:
    st.header("Hi-C Chromatin Explorer")
    hic_output = _resolve_output("hic")
    has_heatmap = (hic_output / "heatmap.html").exists() or (hic_output / "heatmap.png").exists()
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
            st.image(str(hic_output / "heatmap.png"), use_container_width=True)
        if (hic_output / "distance_decay.html").exists():
            with open(hic_output / "distance_decay.html") as f:
                st.components.v1.html(f.read(), height=450, scrolling=False)
        if not qc.empty:
            st.subheader("QC stats (full)")
            st.dataframe(qc, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Run the pipeline first: `python scripts/01_download_data.py` then "
            "`python scripts/02_hic_pipeline.py`"
        )

# --- Nuclei Analyzer ---
with tab3:
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
            st.plotly_chart(fig_area, use_container_width=True)
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
            st.plotly_chart(fig_circ, use_container_width=True)

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
        st.plotly_chart(fig_counts, use_container_width=True)

        st.subheader("Nuclei features (table)")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        if (micro_output / "summary_stats.csv").exists():
            summary = pd.read_csv(micro_output / "summary_stats.csv", index_col=0)
            st.subheader("Summary per image")
            st.dataframe(summary, use_container_width=True, hide_index=True)
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
                st.image(str(sel), use_container_width=True)
            cols = st.columns(min(3, len(overlays)))
            for i, p in enumerate(overlays[:6]):
                cols[i % 3].image(
                    str(p), caption=p.stem[:30] + "...", use_container_width=True
                )
    else:
        st.info(
            "Run the pipeline first: `python scripts/01_download_data.py` then "
            "`python scripts/03_microscopy_pipeline.py`"
        )

# --- Integration ---
with tab4:
    st.header("Morphology–Chromatin Integration")
    micro_output = _resolve_output("microscopy")
    # VLM output written to output/microscopy/; may also exist in deploy_data
    vlm_output_path = ROOT / "output" / "microscopy" / "vlm_output.csv"
    if not vlm_output_path.exists():
        vlm_output_path = micro_output / "vlm_output.csv"
    overlay_dir = micro_output / "overlays"

    # Run VLM Analysis from Streamlit
    st.subheader("VLM Analysis")
    # Use Streamlit secrets if set (server-side key); otherwise show input
    try:
        api_key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key_from_secrets = ""
    api_key = api_key_from_secrets
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key, or add OPENAI_API_KEY to Streamlit secrets for server-side use.",
            key="vlm_api_key",
        )
    else:
        st.caption("Using server-side API key (from Streamlit secrets).")

    # Option 1: Upload your own images
    st.markdown("**Upload images** for VLM phenotype analysis (PNG/JPG recommended):")
    uploaded_files = st.file_uploader(
        "Choose microscopy images (PNG, JPG, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="vlm_upload",
    )
    if st.button("Analyze uploaded images", key="vlm_upload_btn"):
        if not api_key or not str(api_key).strip():
            st.warning("Please enter your OpenAI API key or configure Streamlit secrets.")
        elif not uploaded_files:
            st.warning("Please upload at least one image.")
        else:
            import tempfile
            import importlib.util
            import yaml

            with st.spinner("Analyzing uploaded images..."):
                spec = importlib.util.spec_from_file_location(
                    "vlm_script", ROOT / "scripts" / "05_vlm_analysis.py"
                )
                vlm_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vlm_module)
                with open(ROOT / "config.yaml") as f:
                    vlm_config = yaml.safe_load(f)
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = []
                    for f in uploaded_files[:10]:  # limit to 10
                        p = Path(tmpdir) / f.name
                        p.write_bytes(f.getvalue())
                        paths.append(p)
                    df = vlm_module.analyze_images(
                        paths, str(api_key).strip(), vlm_config
                    )
                if df is not None and len(df) > 0:
                    st.session_state["vlm_uploaded_results"] = df
                    st.success(f"Analyzed {len(df)} image(s).")
                    st.rerun()
                else:
                    st.error("Analysis failed. Check the terminal for details.")

    # Option 2: Run on pipeline overlays (when available)
    if overlay_dir.exists() and list(overlay_dir.glob("*.png")):
        if st.button("Run VLM on pipeline overlays", key="vlm_run_btn"):
            if api_key and str(api_key).strip():
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
                    st.success("VLM analysis complete. Refreshing...")
                    st.rerun()
                else:
                    st.error("VLM analysis failed. Check the terminal for details.")
            else:
                st.warning("Please enter your OpenAI API key.")

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
        st.subheader("VLM Phenotype Descriptions")
        st.caption(
            "Vision-Language Model interpretations provide complementary "
            "phenotype characterization alongside morphology-based strata."
        )
        if "vlm_uploaded_results" in st.session_state:
            if st.button("Clear uploaded results", key="vlm_clear"):
                del st.session_state["vlm_uploaded_results"]
                st.rerun()
        # Table with expandable descriptions
        for _, row in vlm_df.iterrows():
            with st.expander(f"**{row['image']}**"):
                st.write(row["description"])
        # VLM-derived strata (k-means on embeddings)
        embeddings = vlm_df["embedding"].apply(json.loads).tolist()
        if len(embeddings) >= 2:
            n_clusters = min(3, len(embeddings))
            X = np.array(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            vlm_df = vlm_df.copy()
            vlm_df["vlm_cluster"] = kmeans.fit_predict(X)
            cluster_labels = {
                i: f"Cluster {i + 1}"
                for i in range(n_clusters)
            }
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
            st.plotly_chart(fig_vlm, use_container_width=True)
    else:
        st.info(
            "Run `python scripts/05_vlm_analysis.py` to add VLM phenotype analysis. "
            "Requires OPENAI_API_KEY."
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
                st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(
            "Run the microscopy pipeline first: `python scripts/03_microscopy_pipeline.py`"
        )

# --- About ---
with tab5:
    st.header("About")
    st.info(
        "This is an independent demo project and is not affiliated with, "
        "endorsed by, or reviewed by IMBA or the Gerlich lab."
    )
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
