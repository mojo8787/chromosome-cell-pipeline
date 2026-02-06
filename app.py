"""
Chromosome + Cell Mini-Pipeline — Streamlit Dashboard.

Unified interface for Hi-C chromatin exploration and nuclei segmentation analysis.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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

tab1, tab2, tab3 = st.tabs(["Hi-C Explorer", "Nuclei Analyzer", "About"])

# --- Hi-C Explorer ---
with tab1:
    st.header("Hi-C Chromatin Explorer")
    hic_output = ROOT / "output" / "hic"
    if (hic_output / "heatmap.html").exists():
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
        with open(hic_output / "heatmap.html") as f:
            st.components.v1.html(f.read(), height=550, scrolling=False)
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
with tab2:
    st.header("Nuclei Segmentation Analyzer")
    micro_output = ROOT / "output" / "microscopy"
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

# --- About ---
with tab3:
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
            (3, 1, "QC\n(stats, validation)"),
            (5, 1, "Analysis\n(contacts, segmentation)"),
            (7, 1, "Figures & tables\n(heatmaps, overlays)"),
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
