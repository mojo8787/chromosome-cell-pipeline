"""
Chromosome + Cell Mini-Pipeline — Streamlit Dashboard.

Unified interface for Hi-C chromatin exploration and nuclei segmentation analysis.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Chromosome + Cell Pipeline",
    page_icon="🧬",
    layout="wide",
)

st.title("Chromosome + Cell Mini-Pipeline")
st.caption(
    "Hi-C chromatin analysis + microscopy nuclei segmentation | "
    "Compatible with [HiCognition](https://www.hicognition.com) (Gerlich lab)"
)

tab1, tab2, tab3 = st.tabs(["Hi-C Explorer", "Nuclei Analyzer", "About"])

# --- Hi-C Explorer ---
with tab1:
    st.header("Hi-C Chromatin Explorer")
    hic_output = ROOT / "output" / "hic"
    if (hic_output / "heatmap.html").exists():
        with open(hic_output / "heatmap.html") as f:
            st.components.v1.html(f.read(), height=550, scrolling=False)
        if (hic_output / "distance_decay.html").exists():
            with open(hic_output / "distance_decay.html") as f:
                st.components.v1.html(f.read(), height=450, scrolling=False)
        if (hic_output / "qc_stats.csv").exists():
            import pandas as pd

            qc = pd.read_csv(hic_output / "qc_stats.csv")
            st.subheader("QC Stats")
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

        df = pd.read_csv(micro_output / "nuclei_features.csv")
        st.subheader("Nuclei Features")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        if (micro_output / "summary_stats.csv").exists():
            summary = pd.read_csv(micro_output / "summary_stats.csv", index_col=0)
            st.subheader("Summary per Image")
            st.dataframe(summary, use_container_width=True, hide_index=True)
        overlays = list((micro_output / "overlays").glob("*.png"))
        if overlays:
            st.subheader("Sample Overlays")
            cols = st.columns(min(3, len(overlays)))
            for i, p in enumerate(overlays[:6]):
                cols[i % 3].image(str(p), caption=p.name, use_container_width=True)
    else:
        st.info(
            "Run the pipeline first: `python scripts/01_download_data.py` then "
            "`python scripts/03_microscopy_pipeline.py`"
        )

# --- About ---
with tab3:
    st.header("About")
    st.markdown("""
    This demo combines two pipelines:

    1. **Hi-C Genomics** — Load, QC, and visualize Hi-C contact matrices using
       [cooltools](https://cooltools.readthedocs.io/) and the
       [cooler](https://github.com/open2c/cooler) format (same as HiCognition).

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
