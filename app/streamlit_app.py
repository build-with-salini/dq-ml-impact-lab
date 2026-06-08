"""
streamlit_app.py
----------------
Interactive Dashboard for dq-ml-impact-lab.

Pages:
    1. 🏠 Overview          — Project summary and dataset snapshot
    2. 🔍 DQ Profiler        — Upload any CSV → DQ scorecard
    3. ⚗️  Degradation Lab    — Sliders to control degradation → live DQ score
    4. 📉 ML Impact          — DQ score vs model performance chart
    5. 📋 Report             — Downloadable DQ summary

Run with:
    streamlit run app/streamlit_app.py

Author: Salini Anbalagan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import io

# Allow src/ imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dq_profiler import DQProfiler
from src.degrader import DQDegrader
from src.utils import grade_color, format_score

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------

st.set_page_config(
    page_title="dq-ml-impact-lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* Root theme */
    :root {
        --bg-primary: #0d0f14;
        --bg-card: #14171f;
        --bg-elevated: #1a1e29;
        --accent-teal: #00d4aa;
        --accent-orange: #ff6b35;
        --accent-blue: #3d9eff;
        --text-primary: #f0f2f7;
        --text-muted: #8892a4;
        --border: #252b38;
        --grade-a: #2ecc71;
        --grade-b: #a8e063;
        --grade-c: #f39c12;
        --grade-d: #e67e22;
        --grade-f: #e74c3c;
    }

    /* Global overrides */
    .stApp { background-color: var(--bg-primary); }
    .main .block-container { padding: 2rem 2.5rem; max-width: 1300px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-muted);
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
    }

    /* Headers */
    h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
         color: var(--text-primary) !important; letter-spacing: -0.02em; }
    h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
              color: var(--text-primary) !important; }
    p, li { font-family: 'DM Sans', sans-serif !important; color: var(--text-muted) !important; }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--accent-teal);
        line-height: 1;
    }
    .metric-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* Grade badge */
    .grade-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* Section divider */
    .section-tag {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--accent-teal);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.3rem;
    }

    /* Info box */
    .info-box {
        background: var(--bg-elevated);
        border-left: 3px solid var(--accent-teal);
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
    }
    .info-box p { color: var(--text-primary) !important; font-size: 0.9rem !important; margin: 0; }

    /* Streamlit widget tweaks */
    .stSlider > div > div { color: var(--accent-teal) !important; }
    .stSelectbox label, .stSlider label { color: var(--text-muted) !important;
        font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; }
    div[data-testid="stMetric"] label { color: var(--text-muted) !important; }
    div[data-testid="stMetric"] div { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "degraded"
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bank-additional-full.csv"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,23,31,0.8)",
    font=dict(family="DM Sans", color="#8892a4"),
    title_font=dict(family="Syne", color="#f0f2f7", size=15),
    xaxis=dict(gridcolor="#252b38", zerolinecolor="#252b38"),
    yaxis=dict(gridcolor="#252b38", zerolinecolor="#252b38"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f2f7")),
    margin=dict(l=40, r=20, t=50, b=40),
)

MODEL_COLORS = {
    "Logistic Regression": "#3d9eff",
    "Random Forest": "#ff6b35",
}

DEGRADATION_COLORS = {
    "Baseline": "#8892a4",
    "Null Injection": "#e74c3c",
    "Label Noise": "#e67e22",
    "Duplicates": "#3d9eff",
    "Outliers": "#9b59b6",
}

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

@st.cache_data
def load_baseline():
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH, sep=";")
    return None


@st.cache_data
def load_registry():
    path = DATA_DIR / "degradation_registry.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_experiment_results():
    path = DATA_DIR / "experiment_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def run_profiler(df: pd.DataFrame) -> tuple:
    profiler = DQProfiler(df, random_seed=42, contamination=0.05)
    scorecard = profiler.score()
    summary = profiler.summary()
    return scorecard, summary


def get_degradation_type(label: str) -> str:
    if "null" in label:    return "Null Injection"
    if "noise" in label:   return "Label Noise"
    if "dup" in label:     return "Duplicates"
    if "outlier" in label: return "Outliers"
    return "Baseline"


def grade_html(grade: str) -> str:
    color = grade_color(grade)
    return f'<span class="grade-badge" style="background:{color}22;color:{color};border:1px solid {color}44">{grade}</span>'


def metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>
    """


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#f0f2f7'>
            🧪 dq-ml-impact-lab
        </div>
        <div style='font-family:DM Mono,monospace;font-size:0.68rem;color:#00d4aa;
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem'>
            Data Quality × Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔍 DQ Profiler", "⚗️ Degradation Lab", "📉 ML Impact", "📋 Report"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:0.68rem;color:#8892a4;line-height:1.8'>
        <div style='color:#00d4aa;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em'>
            Dataset
        </div>
        UCI Bank Marketing<br>
        ~45k rows · 17 cols<br>
        Binary classification<br><br>
        <div style='color:#00d4aa;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em'>
            Models
        </div>
        Logistic Regression<br>
        Random Forest<br>
        5-fold Stratified CV<br><br>
        <div style='color:#00d4aa;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em'>
            Author
        </div>
        Salini Anbalagan<br>
        Data Steward · Trainer<br>
        PhD Candidate, UKM
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE 1 — OVERVIEW
# ------------------------------------------------------------------

if page == "🏠 Overview":
    st.markdown('<div class="section-tag">dq-ml-impact-lab</div>', unsafe_allow_html=True)
    st.markdown("# Data Quality × ML Performance")
    st.markdown("""
    <p style='font-size:1rem;color:#8892a4;max-width:700px'>
    A reproducible laboratory for measuring how data quality degradation impacts machine learning performance.
    Built on the UCI Bank Marketing dataset — a banking CRM context relevant to fintech and digital banking.
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Stat cards
    df_baseline = load_baseline()
    registry = load_registry()
    exp_results = load_experiment_results()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_rows = f"{len(df_baseline):,}" if df_baseline is not None else "—"
        st.markdown(metric_card("Dataset Rows", n_rows, "UCI Bank Marketing"), unsafe_allow_html=True)
    with c2:
        n_versions = str(len(registry)) if registry is not None else "—"
        st.markdown(metric_card("Dataset Versions", n_versions, "incl. baseline"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Models Trained", "2", "LR + Random Forest"), unsafe_allow_html=True)
    with c4:
        n_exp = str(len(exp_results)) if exp_results is not None else "—"
        st.markdown(metric_card("Experiment Rows", n_exp, "CV results"), unsafe_allow_html=True)

    st.divider()

    # Architecture overview
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### What This Project Does")
        st.markdown("""
        <div class="info-box"><p>
        This lab answers a deceptively simple question:<br>
        <strong style='color:#00d4aa'>How much does dirty data hurt your model?</strong>
        </p></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p>Four types of DQ degradation are applied at increasing severity:</p>
        """, unsafe_allow_html=True)

        degradation_items = [
            ("💧", "Null Injection", "5% · 15% · 30% of values replaced with NaN"),
            ("🏷️", "Label Noise", "5% · 10% · 20% of target labels flipped"),
            ("👯", "Duplicate Rows", "10% · 25% · 50% row inflation"),
            ("📡", "Outlier Injection", "Gaussian noise at σ=2, 3, 4"),
        ]
        for icon, title, desc in degradation_items:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:0.8rem;margin:0.6rem 0;
                        padding:0.7rem 1rem;background:#14171f;border-radius:8px;
                        border:1px solid #252b38'>
                <span style='font-size:1.2rem'>{icon}</span>
                <div>
                    <div style='font-family:DM Sans,sans-serif;color:#f0f2f7;font-weight:500;
                                font-size:0.88rem'>{title}</div>
                    <div style='font-family:DM Mono,monospace;color:#8892a4;font-size:0.72rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Project Pipeline")
        pipeline_steps = [
            ("01", "EDA & Profiling", "Baseline inspection"),
            ("02", "DQ Profiler", "ML-powered scorecard"),
            ("03", "Degradation Engine", "Controlled injection"),
            ("04", "ML Impact Analysis", "Performance curves"),
            ("05", "Dashboard", "You are here"),
        ]
        for num, title, sub in pipeline_steps:
            active = num == "05"
            bg = "#1a1e29" if active else "#14171f"
            border = "#00d4aa" if active else "#252b38"
            color = "#00d4aa" if active else "#8892a4"
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:1rem;margin:0.5rem 0;
                        padding:0.8rem 1rem;background:{bg};border-radius:8px;
                        border:1px solid {border}'>
                <div style='font-family:DM Mono,monospace;color:{color};font-size:0.75rem;
                            font-weight:500;min-width:24px'>{num}</div>
                <div>
                    <div style='font-family:DM Sans,sans-serif;color:#f0f2f7;font-size:0.88rem'>{title}</div>
                    <div style='font-family:DM Mono,monospace;color:#8892a4;font-size:0.7rem'>{sub}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Dataset preview
    if df_baseline is not None:
        st.divider()
        st.markdown("### Dataset Preview")
        st.dataframe(
            df_baseline.head(8),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------------
# PAGE 2 — DQ PROFILER
# ------------------------------------------------------------------

elif page == "🔍 DQ Profiler":
    st.markdown('<div class="section-tag">ML-Powered Scoring</div>', unsafe_allow_html=True)
    st.markdown("# DQ Profiler")
    st.markdown("""
    <p>Upload any CSV dataset and receive a per-column Data Quality scorecard.
    Scores are computed across three dimensions: Completeness, Consistency, and Anomaly Rate.</p>
    """, unsafe_allow_html=True)

    st.divider()

    col_upload, col_or, col_demo = st.columns([2, 0.2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Any tabular CSV. Delimiter is auto-detected.",
        )

    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("▶ Use UCI Demo Dataset", use_container_width=True)

    df_to_profile = None

    if uploaded_file is not None:
        try:
            df_to_profile = pd.read_csv(uploaded_file)
            st.success(f"Loaded: {df_to_profile.shape[0]:,} rows × {df_to_profile.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    elif use_demo:
        df_baseline = load_baseline()
        if df_baseline is not None:
            df_to_profile = df_baseline.copy()
            st.info(f"Using UCI Bank Marketing dataset: {df_to_profile.shape[0]:,} rows × {df_to_profile.shape[1]} columns")
        else:
            st.warning("UCI dataset not found at data/raw/. Run Notebook 01 first to download it.")

    if df_to_profile is not None:
        with st.spinner("Running ML-powered DQ profiling..."):
            scorecard, summary = run_profiler(df_to_profile)

        st.divider()

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(metric_card("Dataset DQ Score",
                        format_score(summary["dataset_dq_score"]),
                        "overall mean"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Completeness",
                        format_score(summary["mean_completeness"]),
                        "mean across cols"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Consistency",
                        format_score(summary["mean_consistency"]),
                        "mean across cols"), unsafe_allow_html=True)
        with c4:
            ar = summary.get("mean_anomaly_rate")
            st.markdown(metric_card("Anomaly Rate",
                        format_score(ar) if ar else "N/A",
                        "numeric cols only"), unsafe_allow_html=True)
        with c5:
            st.markdown(metric_card("At-Risk Columns",
                        str(summary["n_columns_at_risk"]),
                        "score < 0.60"), unsafe_allow_html=True)

        st.divider()

        col_chart, col_table = st.columns([1.3, 1])

        with col_chart:
            st.markdown("#### Overall DQ Score — Per Column")
            fig = go.Figure()
            colors_list = [grade_color(g) for g in scorecard["dq_grade"]]
            fig.add_trace(go.Bar(
                y=scorecard["column"],
                x=scorecard["overall_dq"],
                orientation="h",
                marker_color=colors_list,
                marker_line_width=0,
                text=[f"{v:.3f}" for v in scorecard["overall_dq"]],
                textposition="outside",
                textfont=dict(size=10, color="#8892a4"),
            ))
            for threshold, label, color in [
                (0.85, "A", "#2ecc71"), (0.70, "B", "#a8e063"),
                (0.55, "C", "#f39c12"), (0.40, "D", "#e67e22")
            ]:
                fig.add_vline(x=threshold, line_dash="dot", line_color=color,
                              opacity=0.5, annotation_text=label,
                              annotation_font=dict(color=color, size=10))
            fig.update_layout(**PLOTLY_LAYOUT, height=max(350, len(scorecard) * 30),
                              xaxis_range=[0, 1.12])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("#### Scorecard Table")
            display_sc = scorecard[["column", "dq_grade", "completeness",
                                     "consistency", "anomaly_rate", "overall_dq", "n_nulls"]].copy()
            display_sc["anomaly_rate"] = display_sc["anomaly_rate"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
            st.dataframe(
                display_sc.style.background_gradient(
                    subset=["completeness", "consistency", "overall_dq"],
                    cmap="RdYlGn", vmin=0, vmax=1
                ).format({
                    "completeness": "{:.4f}",
                    "consistency": "{:.4f}",
                    "overall_dq": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # At-risk alert
        at_risk = scorecard[scorecard["overall_dq"] < 0.60]
        if not at_risk.empty:
            st.warning(f"⚠️ {len(at_risk)} column(s) below the 0.60 DQ threshold: "
                       f"{', '.join(at_risk['column'].tolist())}")


# ------------------------------------------------------------------
# PAGE 3 — DEGRADATION LAB
# ------------------------------------------------------------------

elif page == "⚗️ Degradation Lab":
    st.markdown('<div class="section-tag">Interactive Experiment</div>', unsafe_allow_html=True)
    st.markdown("# Degradation Lab")
    st.markdown("""
    <p>Adjust degradation sliders to simulate data quality issues.
    The DQ scorecard updates live to show how each injection type affects quality scores.</p>
    """, unsafe_allow_html=True)

    df_baseline = load_baseline()
    if df_baseline is None:
        st.warning("UCI dataset not found. Run Notebook 01 first.")
        st.stop()

    st.divider()

    # Controls
    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Degradation Controls")

        null_pct = st.slider("💧 Null Injection", 0.0, 0.50, 0.0, 0.05,
                              format="%.0f%%", help="Proportion of values to replace with NaN")
        label_pct = st.slider("🏷️ Label Noise", 0.0, 0.30, 0.0, 0.05,
                               format="%.0f%%", help="Proportion of target labels to flip")
        dup_pct = st.slider("👯 Duplicate Rows", 0.0, 0.60, 0.0, 0.05,
                             format="%.0f%%", help="Proportion of rows to duplicate")
        outlier_sigma = st.slider("📡 Outlier Sigma", 0.0, 5.0, 0.0, 0.5,
                                   help="Standard deviation multiplier for outlier injection (0 = off)")
        outlier_pct = st.slider("📡 Outlier Coverage", 0.01, 0.20, 0.05, 0.01,
                                 format="%.0f%%",
                                 help="Proportion of rows affected by outlier injection") if outlier_sigma > 0 else 0.0

        run_btn = st.button("▶ Run Degradation", use_container_width=True, type="primary")

    with col_results:
        if run_btn or (null_pct + label_pct + dup_pct + outlier_sigma) > 0:
            with st.spinner("Applying degradation and profiling..."):
                degrader = DQDegrader(df_baseline, random_seed=42)

                if null_pct > 0:
                    degrader.inject_nulls(pct=null_pct)
                if label_pct > 0 and "y" in df_baseline.columns:
                    degrader.inject_label_noise(target_col="y", pct=label_pct)
                if dup_pct > 0:
                    degrader.inject_duplicates(pct=dup_pct)
                if outlier_sigma > 0:
                    degrader.inject_outliers(sigma=outlier_sigma, pct=outlier_pct)

                degraded_df = degrader.result()
                scorecard_deg, summary_deg = run_profiler(degraded_df)

                # Also profile baseline for comparison
                _, summary_base = run_profiler(df_baseline)

            # Metrics comparison
            st.markdown("#### Before vs After")
            m1, m2, m3 = st.columns(3)
            for col, label, key in zip([m1, m2, m3],
                                        ["Dataset DQ Score", "Completeness", "Consistency"],
                                        ["dataset_dq_score", "mean_completeness", "mean_consistency"]):
                before = summary_base[key]
                after = summary_deg[key]
                delta = after - before
                col.metric(label, format_score(after),
                           delta=f"{delta:+.4f}",
                           delta_color="inverse")

            # Pipeline log
            pipeline_log = degrader.pipeline_summary()
            if not pipeline_log.empty:
                st.markdown("#### Pipeline Applied")
                for _, row in pipeline_log.iterrows():
                    st.markdown(f"""
                    <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#00d4aa;
                                background:#14171f;padding:0.4rem 0.8rem;border-radius:4px;
                                margin:0.2rem 0;border:1px solid #252b38'>
                        step {row['step']}: {row['operation']}
                    </div>
                    """, unsafe_allow_html=True)

            # Scorecard chart
            st.markdown("#### DQ Scorecard — Degraded")
            fig = go.Figure()
            colors_list = [grade_color(g) for g in scorecard_deg["dq_grade"]]
            fig.add_trace(go.Bar(
                y=scorecard_deg["column"],
                x=scorecard_deg["overall_dq"],
                orientation="h",
                marker_color=colors_list,
                marker_line_width=0,
                name="Degraded",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=max(300, len(scorecard_deg) * 28),
                              xaxis_range=[0, 1.05])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div class="info-box"><p>
            👈 Adjust the sliders on the left and click <strong>Run Degradation</strong>
            to see how DQ scores change in real time.
            </p></div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# PAGE 4 — ML IMPACT
# ------------------------------------------------------------------

elif page == "📉 ML Impact":
    st.markdown('<div class="section-tag">Hero Visualisation</div>', unsafe_allow_html=True)
    st.markdown("# ML Impact Analysis")
    st.markdown("""
    <p>How does data quality degradation affect model performance?
    This page shows the relationship between DQ score and classifier accuracy, F1, and ROC-AUC
    across all degradation versions.</p>
    """, unsafe_allow_html=True)

    exp_results = load_experiment_results()
    registry = load_registry()

    if exp_results is None or registry is None:
        st.warning("Experiment results not found. Run Notebook 04 first to generate results.")
        st.stop()

    # Enrich with DQ scores and type
    dq_lookup = registry[["label", "dataset_dq_score"]].copy()
    results_enriched = exp_results.merge(dq_lookup, on="label", how="left")
    results_enriched["degradation_type"] = results_enriched["label"].apply(get_degradation_type)

    st.divider()

    # Controls
    metric_choice = st.selectbox("Metric", ["f1", "accuracy", "roc_auc"],
                                  format_func=lambda x: x.upper().replace("_", " "))
    filter_type = st.multiselect(
        "Filter by degradation type",
        options=["Baseline", "Null Injection", "Label Noise", "Duplicates", "Outliers"],
        default=["Baseline", "Null Injection", "Label Noise", "Duplicates", "Outliers"],
    )

    filtered = results_enriched[results_enriched["degradation_type"].isin(filter_type)]

    st.divider()

    # Hero scatter
    st.markdown("#### DQ Score vs Model Performance")
    fig = go.Figure()

    for model_name in ["Logistic Regression", "Random Forest"]:
        group = filtered[filtered["model"] == model_name].dropna(
            subset=["dataset_dq_score", metric_choice]
        ).sort_values("dataset_dq_score")

        if group.empty:
            continue

        color = MODEL_COLORS[model_name]
        marker_sym = "circle" if model_name == "Logistic Regression" else "square"

        fig.add_trace(go.Scatter(
            x=group["dataset_dq_score"],
            y=group[metric_choice],
            mode="markers+text",
            name=model_name,
            text=group["label"],
            textposition="top center",
            textfont=dict(size=8, color="#8892a4"),
            marker=dict(size=10, color=color, symbol=marker_sym,
                        line=dict(width=1, color=color)),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "Label: %{text}<br>"
                "DQ Score: %{x:.4f}<br>"
                f"{metric_choice.upper()}: %{{y:.4f}}<extra></extra>"
            )
        ))

        # Trend line
        if len(group) > 2:
            z = np.polyfit(group["dataset_dq_score"], group[metric_choice], 1)
            p = np.poly1d(z)
            x_line = np.linspace(group["dataset_dq_score"].min(),
                                  group["dataset_dq_score"].max(), 60)
            fig.add_trace(go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines", name=f"{model_name} trend",
                line=dict(color=color, dash="dot", width=1.5),
                showlegend=False,
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        xaxis_title="Dataset DQ Score",
        yaxis_title=metric_choice.upper().replace("_", " "),
        title=f"DQ Score vs {metric_choice.upper()} — All Degradation Types",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performance drop table
    st.divider()
    st.markdown("#### Performance Drop vs Baseline")

    pivot = filtered.pivot_table(
        index="label",
        columns="model",
        values=metric_choice,
        aggfunc="first"
    ).reset_index()

    pivot["degradation_type"] = pivot["label"].apply(get_degradation_type)
    pivot = pivot.sort_values("label")

    st.dataframe(
        pivot.style.background_gradient(
            subset=[c for c in pivot.columns if c not in ["label", "degradation_type"]],
            cmap="RdYlGn", vmin=0, vmax=1
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Correlation
    st.divider()
    st.markdown("#### Spearman Correlation: DQ Score vs Performance")

    corr_cols = st.columns(2)
    for i, model_name in enumerate(["Logistic Regression", "Random Forest"]):
        group = results_enriched[results_enriched["model"] == model_name].dropna(
            subset=["dataset_dq_score", metric_choice]
        )
        if len(group) > 2:
            from scipy.stats import spearmanr
            rho, p_val = spearmanr(group["dataset_dq_score"], group[metric_choice])
            sig = "✅ Significant" if p_val < 0.05 else "⚠️ Not significant"
            with corr_cols[i]:
                st.markdown(metric_card(
                    model_name,
                    f"ρ = {rho:.4f}",
                    f"p={p_val:.4f} · {sig}"
                ), unsafe_allow_html=True)


# ------------------------------------------------------------------
# PAGE 5 — REPORT
# ------------------------------------------------------------------

elif page == "📋 Report":
    st.markdown('<div class="section-tag">Export</div>', unsafe_allow_html=True)
    st.markdown("# DQ Report")
    st.markdown("""
    <p>Generate and download a Data Quality summary report for the baseline dataset
    or any uploaded CSV.</p>
    """, unsafe_allow_html=True)

    st.divider()

    col_upload, col_spacer = st.columns([1.5, 1])

    with col_upload:
        report_file = st.file_uploader("Upload CSV (or use baseline below)", type=["csv"])
        use_baseline_report = st.button("▶ Use UCI Baseline", use_container_width=True)

    df_report = None

    if report_file is not None:
        df_report = pd.read_csv(report_file)
        st.success(f"Loaded: {df_report.shape[0]:,} rows × {df_report.shape[1]} columns")
    elif use_baseline_report:
        df_report = load_baseline()
        if df_report is not None:
            st.info("Using UCI Bank Marketing baseline dataset.")

    if df_report is not None:
        with st.spinner("Generating report..."):
            scorecard, summary = run_profiler(df_report)

        st.divider()
        st.markdown("### Summary Statistics")

        summary_df = pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Value": str(v)}
            for k, v in summary.items()
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Full Scorecard")
        st.dataframe(
            scorecard.style.background_gradient(
                subset=["completeness", "consistency", "overall_dq"],
                cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("### Download")

        col_dl1, col_dl2 = st.columns(2)

        # Download scorecard CSV
        with col_dl1:
            csv_buffer = io.StringIO()
            scorecard.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Scorecard CSV",
                data=csv_buffer.getvalue(),
                file_name="dq_scorecard.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Download summary JSON
        with col_dl2:
            import json

            def safe_json(obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if pd.isna(obj) if not isinstance(obj, (list, dict, str)) else False: return None
                return obj

            clean_summary = {k: safe_json(v) for k, v in summary.items()}
            st.download_button(
                label="⬇️ Download Summary JSON",
                data=json.dumps(clean_summary, indent=2),
                file_name="dq_summary.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()
        st.markdown("""
        <div class="info-box"><p>
        <strong style='color:#00d4aa'>Reusability note:</strong>
        Upload any tabular CSV to receive a DQ scorecard.
        The profiler works on any dataset — swap out UCI for your own data.
        </p></div>
        """, unsafe_allow_html=True)
