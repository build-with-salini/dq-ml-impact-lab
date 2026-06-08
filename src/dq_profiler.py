"""
dq_profiler.py
--------------
ML-Powered Data Quality Profiler for dq-ml-impact-lab.

Scores each column across three DQ dimensions:
    - Completeness  : Proportion of non-null values
    - Consistency   : Stability of value distribution (IQR-based for numeric,
                      entropy-based for categorical)
    - Anomaly Rate  : Isolation Forest anomaly detection (numeric columns only)

Produces a per-column DQ Scorecard and a dataset-level summary score.

Designed to be dataset-agnostic. Works on any pandas DataFrame.

Usage:
    from src.dq_profiler import DQProfiler

    profiler = DQProfiler(df, random_seed=42)
    scorecard = profiler.score()
    print(scorecard)

    summary = profiler.summary()
    print(summary)

Author: Salini Anbalagan
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.ensemble import IsolationForest
from scipy.stats import entropy as scipy_entropy
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# DQ dimension weights for computing overall column score
# Equal weighting across three dimensions by default
# Anomaly dimension only applies to numeric columns;
# for categorical columns it is redistributed to the other two.
_WEIGHTS_NUMERIC = {
    "completeness": 1 / 3,
    "consistency": 1 / 3,
    "anomaly_rate": 1 / 3,
}

_WEIGHTS_CATEGORICAL = {
    "completeness": 0.50,
    "consistency": 0.50,
    "anomaly_rate": 0.00,  # not applicable
}


class DQProfiler:
    """
    ML-Powered Data Quality Profiler.

    Scores every column in a DataFrame across three DQ dimensions:
        - Completeness  : How complete is the column? (0 = all null, 1 = no nulls)
        - Consistency   : How stable/expected is the distribution? (0 = highly irregular, 1 = stable)
        - Anomaly Rate  : What proportion of values are anomalous? (0 = all anomalous, 1 = none)

    Overall DQ score per column is the weighted mean of applicable dimensions.
    Dataset-level DQ score is the mean of all column overall scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to profile. Original is never modified.
    random_seed : int, optional
        Seed for Isolation Forest reproducibility. Default is 42.
    contamination : float, optional
        Expected proportion of anomalies for Isolation Forest. Default is 0.05.
        Must be between 0 and 0.5.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        random_seed: int = 42,
        contamination: float = 0.05,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if not (0 < contamination <= 0.5):
            raise ValueError("contamination must be between 0 (exclusive) and 0.5 (inclusive).")

        self._df = df.copy()
        self.random_seed = random_seed
        self.contamination = contamination
        self._scorecard: Optional[pd.DataFrame] = None

        logger.info(
            f"DQProfiler initialised | rows={len(df)} | cols={len(df.columns)} | "
            f"seed={random_seed} | contamination={contamination}"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def score(self) -> pd.DataFrame:
        """
        Compute the full DQ Scorecard for all columns.

        Scores each column across completeness, consistency, and anomaly rate.
        Anomaly rate is computed only for numeric columns via Isolation Forest.

        Returns
        -------
        pd.DataFrame
            DQ Scorecard with columns:
                - column          : Column name
                - dtype           : Column data type
                - completeness    : Completeness score [0, 1]
                - consistency     : Consistency score [0, 1]
                - anomaly_rate    : Anomaly rate score [0, 1] (NaN for categorical)
                - overall_dq      : Weighted overall DQ score [0, 1]
                - dq_grade        : Letter grade (A/B/C/D/F)
                - n_nulls         : Raw null count
                - n_rows          : Total row count

        Example
        -------
        >>> profiler = DQProfiler(df)
        >>> scorecard = profiler.score()
        >>> scorecard.sort_values("overall_dq")
        """
        records = []

        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        # Compute anomaly flags for all numeric columns in one pass (efficient)
        anomaly_flags = self._compute_anomaly_flags(numeric_cols)

        for col in self._df.columns:
            col_series = self._df[col]
            is_numeric = col in numeric_cols
            dtype_name = str(col_series.dtype)

            completeness = self._score_completeness(col_series)
            consistency = self._score_consistency(col_series, is_numeric)

            if is_numeric and anomaly_flags is not None and col in anomaly_flags.columns:
                anomaly_score = self._score_anomaly(anomaly_flags[col])
            else:
                anomaly_score = np.nan

            overall = self._compute_overall(completeness, consistency, anomaly_score, is_numeric)
            grade = self._assign_grade(overall)

            records.append(
                {
                    "column": col,
                    "dtype": dtype_name,
                    "completeness": round(completeness, 4),
                    "consistency": round(consistency, 4),
                    "anomaly_rate": round(anomaly_score, 4) if not np.isnan(anomaly_score) else np.nan,
                    "overall_dq": round(overall, 4),
                    "dq_grade": grade,
                    "n_nulls": int(col_series.isnull().sum()),
                    "n_rows": len(col_series),
                }
            )

        self._scorecard = pd.DataFrame(records).sort_values("overall_dq", ascending=True).reset_index(drop=True)
        logger.info(
            f"Scorecard computed | columns_scored={len(self._scorecard)} | "
            f"dataset_dq={self._scorecard['overall_dq'].mean():.4f}"
        )
        return self._scorecard

    def summary(self) -> Dict[str, float]:
        """
        Return a dataset-level DQ summary.

        Requires score() to have been called first.

        Returns
        -------
        dict
            Keys:
                - dataset_dq_score     : Mean overall DQ score across all columns
                - mean_completeness    : Mean completeness score
                - mean_consistency     : Mean consistency score
                - mean_anomaly_rate    : Mean anomaly rate score (numeric cols only)
                - n_columns            : Total columns profiled
                - n_columns_at_risk    : Columns with overall_dq < 0.60
                - worst_column         : Column with lowest overall_dq
                - best_column          : Column with highest overall_dq

        Example
        -------
        >>> profiler.score()
        >>> profiler.summary()
        """
        if self._scorecard is None:
            raise RuntimeError("Call score() before summary().")

        sc = self._scorecard
        numeric_rows = sc[sc["anomaly_rate"].notna()]

        worst = sc.iloc[0]["column"]
        best = sc.iloc[-1]["column"]
        at_risk = int((sc["overall_dq"] < 0.60).sum())

        summary = {
            "dataset_dq_score": round(sc["overall_dq"].mean(), 4),
            "mean_completeness": round(sc["completeness"].mean(), 4),
            "mean_consistency": round(sc["consistency"].mean(), 4),
            "mean_anomaly_rate": round(numeric_rows["anomaly_rate"].mean(), 4) if not numeric_rows.empty else np.nan,
            "n_columns": len(sc),
            "n_columns_at_risk": at_risk,
            "worst_column": worst,
            "best_column": best,
        }

        logger.info(f"Summary | dataset_dq={summary['dataset_dq_score']} | at_risk={at_risk}")
        return summary

    def worst_columns(self, n: int = 5) -> pd.DataFrame:
        """
        Return the n columns with the lowest overall DQ score.

        Parameters
        ----------
        n : int
            Number of worst columns to return. Default is 5.

        Returns
        -------
        pd.DataFrame
            Subset of scorecard sorted by overall_dq ascending.

        Example
        -------
        >>> profiler.score()
        >>> profiler.worst_columns(n=3)
        """
        if self._scorecard is None:
            raise RuntimeError("Call score() before worst_columns().")
        return self._scorecard.head(n)

    def at_risk_columns(self, threshold: float = 0.60) -> pd.DataFrame:
        """
        Return all columns with overall_dq below a given threshold.

        Parameters
        ----------
        threshold : float
            DQ score below which a column is considered at risk. Default is 0.60.

        Returns
        -------
        pd.DataFrame
            Filtered scorecard of at-risk columns.

        Example
        -------
        >>> profiler.score()
        >>> profiler.at_risk_columns(threshold=0.70)
        """
        if self._scorecard is None:
            raise RuntimeError("Call score() before at_risk_columns().")
        return self._scorecard[self._scorecard["overall_dq"] < threshold].copy()

    # ------------------------------------------------------------------
    # DIMENSION SCORERS
    # ------------------------------------------------------------------

    def _score_completeness(self, series: pd.Series) -> float:
        """
        Completeness = proportion of non-null values.
        Score of 1.0 means fully complete; 0.0 means entirely null.
        """
        if len(series) == 0:
            return 0.0
        return 1.0 - series.isnull().mean()

    def _score_consistency(self, series: pd.Series, is_numeric: bool) -> float:
        """
        Consistency score:
          - Numeric  : IQR-based spread ratio. Narrow spread = high consistency.
          - Categorical : Normalised entropy. Low entropy = high consistency
                          (one dominant value = consistent).

        Both return a score in [0, 1].
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return 0.0

        if is_numeric:
            return self._numeric_consistency(non_null)
        else:
            return self._categorical_consistency(non_null)

    def _numeric_consistency(self, series: pd.Series) -> float:
        """
        IQR / range ratio as a spread measure.
        A tight IQR relative to the full range indicates consistency.
        Score = 1 - (IQR / range), clipped to [0, 1].
        Falls back to 1.0 if range is zero (all values identical = perfectly consistent).
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        value_range = series.max() - series.min()

        if value_range == 0:
            return 1.0  # All values identical — perfectly consistent

        spread_ratio = iqr / value_range
        # High spread_ratio means IQR is wide relative to range — less consistent
        # We invert: high consistency = low spread ratio
        consistency = 1.0 - spread_ratio
        return float(np.clip(consistency, 0.0, 1.0))

    def _categorical_consistency(self, series: pd.Series) -> float:
        """
        Normalised Shannon entropy of value frequencies.
        Low entropy = one or few dominant values = high consistency.
        Score = 1 - normalised_entropy, so 1.0 = perfectly consistent.
        """
        value_counts = series.value_counts(normalize=True)
        n_categories = len(value_counts)

        if n_categories == 1:
            return 1.0  # Only one unique value — perfectly consistent

        raw_entropy = scipy_entropy(value_counts)
        max_entropy = np.log(n_categories)  # Maximum possible entropy for n categories
        normalised = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        return float(np.clip(1.0 - normalised, 0.0, 1.0))

    def _compute_anomaly_flags(
        self, numeric_cols: list
    ) -> Optional[pd.DataFrame]:
        """
        Run Isolation Forest across all numeric columns simultaneously.
        Returns a DataFrame of boolean anomaly flags (True = anomaly).
        Returns None if no numeric columns exist or data is insufficient.
        """
        if not numeric_cols:
            logger.warning("No numeric columns found. Skipping anomaly detection.")
            return None

        numeric_df = self._df[numeric_cols].copy()
        # Fill nulls with column median for Isolation Forest (nulls not supported)
        numeric_df = numeric_df.fillna(numeric_df.median())

        if len(numeric_df) < 10:
            logger.warning(
                "Fewer than 10 rows available for Isolation Forest. Skipping anomaly detection."
            )
            return None

        try:
            iso = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_seed,
                n_jobs=-1,
            )
            predictions = iso.fit_predict(numeric_df)
            # IsolationForest returns -1 for anomalies, 1 for inliers
            anomaly_flags = pd.DataFrame(
                predictions == -1,
                columns=numeric_cols,
                index=numeric_df.index,
            )
            logger.info(
                f"Isolation Forest complete | numeric_cols={len(numeric_cols)} | "
                f"anomalies_detected={int((predictions == -1).sum())}"
            )
            return anomaly_flags

        except Exception as e:
            logger.error(f"Isolation Forest failed: {e}")
            return None

    def _score_anomaly(self, anomaly_flag_col: pd.Series) -> float:
        """
        Anomaly rate score = 1 - proportion of anomalous rows.
        Score of 1.0 means no anomalies detected; 0.0 means all rows anomalous.
        """
        if len(anomaly_flag_col) == 0:
            return 1.0
        anomaly_proportion = anomaly_flag_col.mean()
        return float(np.clip(1.0 - anomaly_proportion, 0.0, 1.0))

    # ------------------------------------------------------------------
    # OVERALL SCORE + GRADE
    # ------------------------------------------------------------------

    def _compute_overall(
        self,
        completeness: float,
        consistency: float,
        anomaly_score: float,
        is_numeric: bool,
    ) -> float:
        """
        Compute weighted overall DQ score from dimension scores.
        Uses numeric weights if anomaly score is available,
        categorical weights (no anomaly dimension) otherwise.
        """
        if is_numeric and not np.isnan(anomaly_score):
            w = _WEIGHTS_NUMERIC
            overall = (
                w["completeness"] * completeness
                + w["consistency"] * consistency
                + w["anomaly_rate"] * anomaly_score
            )
        else:
            w = _WEIGHTS_CATEGORICAL
            overall = (
                w["completeness"] * completeness
                + w["consistency"] * consistency
            )
        return float(np.clip(overall, 0.0, 1.0))

    def _assign_grade(self, score: float) -> str:
        """
        Assign a letter grade based on overall DQ score.

        Grading scale:
            A  : >= 0.85  (Excellent)
            B  : >= 0.70  (Good)
            C  : >= 0.55  (Acceptable)
            D  : >= 0.40  (Poor)
            F  : <  0.40  (Critical)
        """
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        else:
            return "F"

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def scorecard(self) -> Optional[pd.DataFrame]:
        """Return the most recently computed scorecard, or None if not yet run."""
        return self._scorecard

    @property
    def dataset_dq_score(self) -> Optional[float]:
        """Return the dataset-level mean DQ score, or None if not yet computed."""
        if self._scorecard is None:
            return None
        return round(self._scorecard["overall_dq"].mean(), 4)

    def __repr__(self) -> str:
        scored = self._scorecard is not None
        score_str = f"{self.dataset_dq_score:.4f}" if scored else "not computed"
        return (
            f"DQProfiler("
            f"rows={len(self._df)}, "
            f"cols={len(self._df.columns)}, "
            f"dataset_dq={score_str}, "
            f"seed={self.random_seed})"
        )
