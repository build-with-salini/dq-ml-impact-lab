"""
utils.py
--------
Shared Utility Functions for dq-ml-impact-lab.

Provides helpers used across notebooks, src modules, and the Streamlit app:
    - Data loading and validation
    - Experiment result formatting
    - Plot-ready data preparation
    - File I/O helpers
    - Logging setup

Usage:
    from src.utils import load_dataset, results_to_plot_df, save_scorecard

Author: Salini Anbalagan
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently formatted logger for any module.

    Parameters
    ----------
    name : str
        Logger name — typically __name__ from the calling module.
    level : int
        Logging level. Default is logging.INFO.

    Returns
    -------
    logging.Logger

    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.info("Module loaded.")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------

def load_dataset(
    path: str,
    sep: str = ";",
    target_col: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV dataset from disk with basic validation.

    Defaults to semicolon delimiter to match the UCI Bank Marketing
    dataset format. Override sep for other datasets.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    sep : str
        Column delimiter. Default is ";" (UCI Bank Marketing format).
    target_col : str, optional
        If provided, validates that the target column exists after loading.
    verbose : bool
        If True, logs shape and basic info. Default is True.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If target_col is specified but not found in the loaded DataFrame.

    Example
    -------
    >>> df = load_dataset("data/raw/bank-additional-full.csv", target_col="y")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, sep=sep)

    if target_col and target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if verbose:
        logger.info(
            f"Dataset loaded | path={path} | shape={df.shape} | "
            f"nulls={df.isnull().sum().sum()}"
        )
        if target_col:
            dist = df[target_col].value_counts(normalize=True).round(3).to_dict()
            logger.info(f"Target distribution | {dist}")

    return df


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 10,
    required_cols: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame against basic quality gates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    min_rows : int
        Minimum number of rows required. Default is 10.
    required_cols : list of str, optional
        Column names that must be present.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
        is_valid is True only if no issues are found.

    Example
    -------
    >>> is_valid, issues = validate_dataframe(df, required_cols=["age", "y"])
    >>> if not is_valid:
    ...     print(issues)
    """
    issues = []

    if not isinstance(df, pd.DataFrame):
        return False, ["Input is not a pandas DataFrame."]

    if df.empty:
        issues.append("DataFrame is empty.")

    if len(df) < min_rows:
        issues.append(f"DataFrame has fewer than {min_rows} rows ({len(df)} found).")

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            issues.append(f"Required columns missing: {missing}")

    return len(issues) == 0, issues


# ------------------------------------------------------------------
# EXPERIMENT RESULT FORMATTING
# ------------------------------------------------------------------

def results_to_plot_df(
    results: pd.DataFrame,
    metric: str = "f1",
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reshape experiment results into a plot-ready long-format DataFrame.

    Selects a single metric and optionally filters by model.
    Output is suitable for direct use with matplotlib, seaborn, or Plotly.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ModelTrainer.run_degradation_experiment().
        Must contain columns: label, model, and the specified metric.
    metric : str
        Metric column to extract. Default is "f1".
    models : list of str, optional
        If provided, filters to these model names only.

    Returns
    -------
    pd.DataFrame
        Columns: label, model, {metric}, {metric}_std

    Raises
    ------
    ValueError
        If metric column is not found in results.

    Example
    -------
    >>> plot_df = results_to_plot_df(results, metric="accuracy")
    """
    if metric not in results.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results. "
            f"Available: {[c for c in results.columns if c not in ['label', 'model']]}"
        )

    std_col = f"{metric}_std"
    cols = ["label", "model", metric]
    if std_col in results.columns:
        cols.append(std_col)

    plot_df = results[cols].copy()

    if models:
        plot_df = plot_df[plot_df["model"].isin(models)]

    return plot_df.reset_index(drop=True)


def compute_performance_delta(
    results: pd.DataFrame,
    metric: str = "f1",
    baseline_label: str = "baseline",
) -> pd.DataFrame:
    """
    Compute the performance drop relative to baseline for each degradation level.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ModelTrainer.run_degradation_experiment().
    metric : str
        Metric to compute delta for. Default is "f1".
    baseline_label : str
        Label used for the clean baseline run. Default is "baseline".

    Returns
    -------
    pd.DataFrame
        Original results with two additional columns:
            - {metric}_delta     : Absolute change from baseline (negative = drop)
            - {metric}_pct_drop  : Percentage drop from baseline

    Example
    -------
    >>> results_with_delta = compute_performance_delta(results, metric="f1")
    """
    if metric not in results.columns:
        raise ValueError(f"Metric '{metric}' not found in results.")

    results = results.copy()
    results[f"{metric}_delta"] = np.nan
    results[f"{metric}_pct_drop"] = np.nan

    for model in results["model"].unique():
        mask = results["model"] == model
        baseline_row = results[mask & (results["label"] == baseline_label)]

        if baseline_row.empty:
            logger.warning(
                f"No baseline row found for model='{model}'. Skipping delta computation."
            )
            continue

        baseline_score = baseline_row[metric].values[0]

        if pd.isna(baseline_score) or baseline_score == 0:
            continue

        results.loc[mask, f"{metric}_delta"] = (
            results.loc[mask, metric] - baseline_score
        )
        results.loc[mask, f"{metric}_pct_drop"] = (
            (results.loc[mask, metric] - baseline_score) / baseline_score * 100
        ).round(2)

    return results


def summarise_experiment(
    results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Produce a concise summary table of experiment results.

    Shows mean metric scores per degradation label, pivoted for readability.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ModelTrainer.run_degradation_experiment().
    metrics : list of str, optional
        Metrics to include. Defaults to ["accuracy", "f1", "roc_auc"].

    Returns
    -------
    pd.DataFrame
        Summary table with label as index, model+metric as columns.

    Example
    -------
    >>> summary = summarise_experiment(results)
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "roc_auc"]

    available = [m for m in metrics if m in results.columns]
    if not available:
        raise ValueError(f"None of the requested metrics found in results: {metrics}")

    summary = results[["label", "model"] + available].copy()
    summary = summary.set_index(["label", "model"])
    return summary


# ------------------------------------------------------------------
# FILE I/O
# ------------------------------------------------------------------

def save_scorecard(
    scorecard: pd.DataFrame,
    output_path: str = "data/degraded/scorecard.csv",
    verbose: bool = True,
) -> str:
    """
    Save a DQ scorecard DataFrame to CSV.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    scorecard : pd.DataFrame
        DQ scorecard from DQProfiler.score().
    output_path : str
        Output file path. Default is "data/degraded/scorecard.csv".
    verbose : bool
        If True, logs the save path and shape.

    Returns
    -------
    str
        Resolved absolute path of the saved file.

    Example
    -------
    >>> save_scorecard(scorecard, "data/degraded/baseline_scorecard.csv")
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(path, index=False)

    if verbose:
        logger.info(f"Scorecard saved | path={path.resolve()} | shape={scorecard.shape}")

    return str(path.resolve())


def save_experiment_results(
    results: pd.DataFrame,
    output_path: str = "data/degraded/experiment_results.csv",
    verbose: bool = True,
) -> str:
    """
    Save experiment results DataFrame to CSV.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ModelTrainer.run_degradation_experiment().
    output_path : str
        Output file path.
    verbose : bool
        If True, logs the save path.

    Returns
    -------
    str
        Resolved absolute path of the saved file.

    Example
    -------
    >>> save_experiment_results(results, "data/degraded/results.csv")
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)

    if verbose:
        logger.info(f"Results saved | path={path.resolve()} | rows={len(results)}")

    return str(path.resolve())


def save_summary(
    summary: Dict[str, Any],
    output_path: str = "data/degraded/dq_summary.json",
    verbose: bool = True,
) -> str:
    """
    Save a DQ summary dictionary to JSON.

    Parameters
    ----------
    summary : dict
        Output from DQProfiler.summary().
    output_path : str
        Output file path.
    verbose : bool
        If True, logs the save path.

    Returns
    -------
    str
        Resolved absolute path of the saved file.

    Example
    -------
    >>> save_summary(profiler.summary(), "data/degraded/summary.json")
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python for JSON serialisation
    clean_summary = _sanitise_for_json(summary)

    with open(path, "w") as f:
        json.dump(clean_summary, f, indent=2)

    if verbose:
        logger.info(f"Summary saved | path={path.resolve()}")

    return str(path.resolve())


def load_experiment_results(path: str) -> pd.DataFrame:
    """
    Load previously saved experiment results from CSV.

    Parameters
    ----------
    path : str
        Path to the saved results CSV.

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> results = load_experiment_results("data/degraded/experiment_results.csv")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")
    df = pd.read_csv(p)
    logger.info(f"Results loaded | path={p} | rows={len(df)}")
    return df


# ------------------------------------------------------------------
# DISPLAY HELPERS
# ------------------------------------------------------------------

def grade_color(grade: str) -> str:
    """
    Map a DQ grade letter to a hex colour string for display.

    Used in Streamlit dashboard and notebook styling.

    Parameters
    ----------
    grade : str
        One of "A", "B", "C", "D", "F".

    Returns
    -------
    str
        Hex colour code.

    Example
    -------
    >>> grade_color("A")
    '#2ecc71'
    """
    mapping = {
        "A": "#2ecc71",   # Green
        "B": "#a8e063",   # Light green
        "C": "#f39c12",   # Amber
        "D": "#e67e22",   # Orange
        "F": "#e74c3c",   # Red
    }
    return mapping.get(grade.upper(), "#95a5a6")


def format_score(score: float, as_pct: bool = False) -> str:
    """
    Format a DQ score float for display.

    Parameters
    ----------
    score : float
        Score value between 0 and 1.
    as_pct : bool
        If True, returns as percentage string (e.g., "87.3%").
        If False, returns as decimal string (e.g., "0.873").

    Returns
    -------
    str

    Example
    -------
    >>> format_score(0.873)
    '0.8730'
    >>> format_score(0.873, as_pct=True)
    '87.3%'
    """
    if pd.isna(score):
        return "N/A"
    if as_pct:
        return f"{score * 100:.1f}%"
    return f"{score:.4f}"


def print_scorecard(scorecard: pd.DataFrame, top_n: Optional[int] = None) -> None:
    """
    Pretty-print a DQ scorecard to stdout.

    Parameters
    ----------
    scorecard : pd.DataFrame
        DQ scorecard from DQProfiler.score().
    top_n : int, optional
        If provided, print only the top_n worst columns.

    Example
    -------
    >>> print_scorecard(scorecard, top_n=5)
    """
    display_cols = ["column", "dtype", "completeness", "consistency", "anomaly_rate", "overall_dq", "dq_grade"]
    available = [c for c in display_cols if c in scorecard.columns]

    df_display = scorecard[available].copy()
    if top_n:
        df_display = df_display.head(top_n)

    print("\n" + "=" * 70)
    print(f"  DQ SCORECARD {'(worst ' + str(top_n) + ' columns)' if top_n else '(all columns)'}")
    print("=" * 70)
    print(df_display.to_string(index=False))
    print("=" * 70 + "\n")


# ------------------------------------------------------------------
# PRIVATE HELPERS
# ------------------------------------------------------------------

def _sanitise_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python for JSON serialisation.
    """
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif pd.isna(obj) if not isinstance(obj, (list, dict, str)) else False:
        return None
    return obj
