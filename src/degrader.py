"""
degrader.py
-----------
Controlled Data Quality Degradation Engine for dq-ml-impact-lab.

Provides both:
  - Independent degradation methods (inject one type at a time)
  - A pipeline composer (chain multiple degradations reproducibly)

Designed to be dataset-agnostic. Works on any pandas DataFrame.

Usage:
    from src.degrader import DQDegrader

    degrader = DQDegrader(df, random_seed=42)

    # Independent method
    degraded_df = degrader.inject_nulls(columns=["age", "balance"], pct=0.15)

    # Pipeline composer
    degraded_df = (
        DQDegrader(df, random_seed=42)
        .inject_nulls(pct=0.10)
        .inject_label_noise(target_col="y", pct=0.05)
        .inject_duplicates(pct=0.20)
        .inject_outliers(columns=["balance"], sigma=3.0)
        .result()
    )

Author: Salini Anbalagan
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DQDegrader:
    """
    Controlled Data Quality Degradation Engine.

    Injects realistic DQ issues into a DataFrame for ML impact experiments.
    All operations are reproducible via a fixed random seed.

    Supported degradation types:
        - Null injection       : Randomly replace values with NaN
        - Label noise          : Randomly flip binary target labels
        - Duplicate injection  : Inflate dataset with repeated rows
        - Outlier injection    : Add statistical outliers via Gaussian noise

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to degrade. Original is never modified.
    random_seed : int, optional
        Seed for all random operations. Default is 42.
    """

    def __init__(self, df: pd.DataFrame, random_seed: int = 42):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        self._original = df.copy()
        self._working = df.copy()
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self._pipeline_log: List[str] = []

        logger.info(
            f"DQDegrader initialised | rows={len(df)} | cols={len(df.columns)} | seed={random_seed}"
        )

    # ------------------------------------------------------------------
    # INDEPENDENT METHODS
    # ------------------------------------------------------------------

    def inject_nulls(
        self,
        pct: float,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> Union["DQDegrader", pd.DataFrame]:
        """
        Randomly replace values with NaN to simulate missingness.

        Parameters
        ----------
        pct : float
            Proportion of values to nullify per column. Must be between 0 and 1.
        columns : list of str, optional
            Columns to target. Defaults to all columns if None.
        inplace : bool
            If True, returns a DataFrame directly. If False, returns self for chaining.

        Returns
        -------
        DQDegrader (for chaining) or pd.DataFrame (if inplace=True)

        Example
        -------
        >>> degrader.inject_nulls(pct=0.15, columns=["age", "balance"])
        """
        self._validate_pct(pct, "inject_nulls")
        target_cols = self._resolve_columns(columns, exclude_dtypes=None)

        df = self._working.copy()
        n_rows = len(df)
        n_nulls = int(n_rows * pct)

        for col in target_cols:
            null_indices = self._rng.choice(n_rows, size=n_nulls, replace=False)
            df.loc[null_indices, col] = np.nan

        null_count = df[target_cols].isnull().sum().sum()
        logger.info(
            f"inject_nulls | pct={pct:.0%} | cols={len(target_cols)} | "
            f"total_nulls_injected={null_count}"
        )
        self._pipeline_log.append(
            f"inject_nulls(pct={pct}, cols={target_cols})"
        )
        self._working = df

        if inplace:
            return self._working.copy()
        return self

    def inject_label_noise(
        self,
        target_col: str,
        pct: float,
        inplace: bool = False,
    ) -> Union["DQDegrader", pd.DataFrame]:
        """
        Randomly flip binary target labels to simulate annotation noise.

        Parameters
        ----------
        target_col : str
            Name of the binary target column.
        pct : float
            Proportion of labels to flip. Must be between 0 and 1.
        inplace : bool
            If True, returns a DataFrame directly. If False, returns self for chaining.

        Returns
        -------
        DQDegrader (for chaining) or pd.DataFrame (if inplace=True)

        Notes
        -----
        Assumes binary labels (e.g., 0/1 or "yes"/"no").
        For multi-class targets, only two-class flipping is supported.

        Example
        -------
        >>> degrader.inject_label_noise(target_col="y", pct=0.10)
        """
        self._validate_pct(pct, "inject_label_noise")

        if target_col not in self._working.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")

        df = self._working.copy()
        n_rows = len(df)
        n_flips = int(n_rows * pct)

        unique_labels = df[target_col].dropna().unique()
        if len(unique_labels) != 2:
            raise ValueError(
                f"inject_label_noise supports binary targets only. "
                f"Found {len(unique_labels)} unique values in '{target_col}'."
            )

        flip_indices = self._rng.choice(n_rows, size=n_flips, replace=False)
        label_map = {unique_labels[0]: unique_labels[1], unique_labels[1]: unique_labels[0]}
        df.loc[flip_indices, target_col] = df.loc[flip_indices, target_col].map(label_map)

        logger.info(
            f"inject_label_noise | target='{target_col}' | pct={pct:.0%} | "
            f"labels_flipped={n_flips}"
        )
        self._pipeline_log.append(
            f"inject_label_noise(target_col='{target_col}', pct={pct})"
        )
        self._working = df

        if inplace:
            return self._working.copy()
        return self

    def inject_duplicates(
        self,
        pct: float,
        inplace: bool = False,
    ) -> Union["DQDegrader", pd.DataFrame]:
        """
        Inflate the dataset by duplicating a proportion of rows.

        Parameters
        ----------
        pct : float
            Proportion of original rows to duplicate. A value of 0.20
            adds rows equal to 20% of the original dataset size.
        inplace : bool
            If True, returns a DataFrame directly. If False, returns self for chaining.

        Returns
        -------
        DQDegrader (for chaining) or pd.DataFrame (if inplace=True)

        Example
        -------
        >>> degrader.inject_duplicates(pct=0.25)
        """
        self._validate_pct(pct, "inject_duplicates")

        df = self._working.copy()
        n_rows = len(df)
        n_dupes = int(n_rows * pct)

        dupe_indices = self._rng.choice(n_rows, size=n_dupes, replace=True)
        duped_rows = df.iloc[dupe_indices].copy()
        df = pd.concat([df, duped_rows], ignore_index=True)

        logger.info(
            f"inject_duplicates | pct={pct:.0%} | rows_added={n_dupes} | "
            f"new_total={len(df)}"
        )
        self._pipeline_log.append(f"inject_duplicates(pct={pct})")
        self._working = df

        if inplace:
            return self._working.copy()
        return self

    def inject_outliers(
        self,
        columns: Optional[List[str]] = None,
        sigma: float = 3.0,
        pct: float = 0.05,
        inplace: bool = False,
    ) -> Union["DQDegrader", pd.DataFrame]:
        """
        Inject statistical outliers into numeric columns via Gaussian noise.

        Parameters
        ----------
        columns : list of str, optional
            Numeric columns to target. Defaults to all numeric columns.
        sigma : float
            Standard deviation multiplier for outlier magnitude. Default is 3.0.
        pct : float
            Proportion of rows to affect per column. Default is 0.05.
        inplace : bool
            If True, returns a DataFrame directly. If False, returns self for chaining.

        Returns
        -------
        DQDegrader (for chaining) or pd.DataFrame (if inplace=True)

        Example
        -------
        >>> degrader.inject_outliers(columns=["balance", "duration"], sigma=3.5, pct=0.05)
        """
        self._validate_pct(pct, "inject_outliers")

        target_cols = self._resolve_columns(columns, exclude_dtypes=["object", "category"])
        if not target_cols:
            raise ValueError("No numeric columns found for outlier injection.")

        df = self._working.copy()
        n_rows = len(df)
        n_outliers = int(n_rows * pct)

        for col in target_cols:
            col_std = df[col].std()
            col_mean = df[col].mean()
            outlier_indices = self._rng.choice(n_rows, size=n_outliers, replace=False)
            noise = self._rng.normal(loc=0, scale=sigma * col_std, size=n_outliers)
            df.loc[outlier_indices, col] = col_mean + noise

        logger.info(
            f"inject_outliers | sigma={sigma} | pct={pct:.0%} | "
            f"cols={target_cols} | rows_affected_per_col={n_outliers}"
        )
        self._pipeline_log.append(
            f"inject_outliers(cols={target_cols}, sigma={sigma}, pct={pct})"
        )
        self._working = df

        if inplace:
            return self._working.copy()
        return self

    # ------------------------------------------------------------------
    # PIPELINE COMPOSER
    # ------------------------------------------------------------------

    def result(self) -> pd.DataFrame:
        """
        Return the final degraded DataFrame after all pipeline steps.

        Use at the end of a method chain.

        Returns
        -------
        pd.DataFrame
            The degraded DataFrame with all pipeline operations applied.

        Example
        -------
        >>> degraded_df = (
        ...     DQDegrader(df, random_seed=42)
        ...     .inject_nulls(pct=0.10)
        ...     .inject_label_noise(target_col="y", pct=0.05)
        ...     .result()
        ... )
        """
        logger.info(
            f"Pipeline complete | steps={len(self._pipeline_log)} | "
            f"final_shape={self._working.shape}"
        )
        return self._working.copy()

    def pipeline_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of all degradation steps applied in this session.

        Returns
        -------
        pd.DataFrame
            Step-by-step log of pipeline operations.

        Example
        -------
        >>> degrader.pipeline_summary()
        """
        summary = pd.DataFrame(
            {
                "step": range(1, len(self._pipeline_log) + 1),
                "operation": self._pipeline_log,
            }
        )
        return summary

    def reset(self) -> "DQDegrader":
        """
        Reset the working DataFrame back to the original clean state.
        Clears the pipeline log and re-initialises the RNG.

        Returns
        -------
        DQDegrader
            Self, for optional chaining after reset.

        Example
        -------
        >>> degrader.reset().inject_nulls(pct=0.30)
        """
        self._working = self._original.copy()
        self._rng = np.random.default_rng(self.random_seed)
        self._pipeline_log = []
        logger.info("DQDegrader reset to original state.")
        return self

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def original(self) -> pd.DataFrame:
        """Return a copy of the original unmodified DataFrame."""
        return self._original.copy()

    @property
    def shape(self) -> tuple:
        """Return the current shape of the working DataFrame."""
        return self._working.shape

    @property
    def null_rate(self) -> float:
        """Return the current overall null rate of the working DataFrame."""
        total_cells = self._working.size
        null_cells = self._working.isnull().sum().sum()
        return round(null_cells / total_cells, 4) if total_cells > 0 else 0.0

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _validate_pct(self, pct: float, method_name: str) -> None:
        """Validate that pct is a float strictly between 0 and 1."""
        if not isinstance(pct, (float, int)) or not (0 < pct < 1):
            raise ValueError(
                f"{method_name}: 'pct' must be a float strictly between 0 and 1. Got: {pct}"
            )

    def _resolve_columns(
        self,
        columns: Optional[List[str]],
        exclude_dtypes: Optional[List[str]],
    ) -> List[str]:
        """
        Resolve target column list.
        If columns is None, defaults to all columns (optionally filtered by dtype).
        """
        if columns is not None:
            missing = [c for c in columns if c not in self._working.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            return columns

        if exclude_dtypes:
            return [
                c for c in self._working.columns
                if self._working[c].dtype.name not in exclude_dtypes
            ]
        return list(self._working.columns)

    def __repr__(self) -> str:
        return (
            f"DQDegrader("
            f"rows={self.shape[0]}, "
            f"cols={self.shape[1]}, "
            f"null_rate={self.null_rate:.2%}, "
            f"seed={self.random_seed}, "
            f"pipeline_steps={len(self._pipeline_log)})"
        )
