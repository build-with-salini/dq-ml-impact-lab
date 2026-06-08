"""
test_degrader.py
----------------
Unit Tests for DQDegrader (src/degrader.py).

Covers:
    - Initialisation validation
    - inject_nulls         : null count, scope, reproducibility
    - inject_label_noise   : label flip count, binary constraint
    - inject_duplicates    : row count inflation
    - inject_outliers      : numeric-only targeting, outlier magnitude
    - Pipeline composer    : chaining, result(), reset(), pipeline_summary()
    - Properties           : null_rate, shape, original
    - Edge cases           : empty DataFrame, invalid pct, missing columns

Run with:
    pytest tests/test_degrader.py -v

Author: Salini Anbalagan
"""

import pytest
import pandas as pd
import numpy as np

from src.degrader import DQDegrader


# ------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------

@pytest.fixture
def clean_df():
    """
    A small, clean DataFrame simulating a banking dataset structure.
    100 rows, mixed numeric and categorical columns, no nulls.
    """
    np.random.seed(0)
    n = 100
    return pd.DataFrame(
        {
            "age":      np.random.randint(18, 70, size=n),
            "balance":  np.random.randint(-500, 10000, size=n),
            "duration": np.random.randint(0, 600, size=n),
            "job":      np.random.choice(["admin", "technician", "services"], size=n),
            "marital":  np.random.choice(["married", "single", "divorced"], size=n),
            "y":        np.random.choice(["yes", "no"], size=n),
        }
    )


@pytest.fixture
def degrader(clean_df):
    """Default DQDegrader instance with seed=42."""
    return DQDegrader(clean_df, random_seed=42)


# ------------------------------------------------------------------
# INITIALISATION
# ------------------------------------------------------------------

class TestInitialisation:

    def test_valid_init(self, clean_df):
        d = DQDegrader(clean_df, random_seed=42)
        assert d.shape == clean_df.shape
        assert d.random_seed == 42

    def test_original_is_copy(self, clean_df):
        """Modifying the input DataFrame should not affect DQDegrader internals."""
        d = DQDegrader(clean_df, random_seed=42)
        clean_df["age"] = -999
        assert (d.original["age"] != -999).all()

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            DQDegrader([1, 2, 3])

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            DQDegrader(pd.DataFrame())

    def test_repr(self, degrader):
        r = repr(degrader)
        assert "DQDegrader" in r
        assert "seed=42" in r


# ------------------------------------------------------------------
# INJECT NULLS
# ------------------------------------------------------------------

class TestInjectNulls:

    def test_null_count_increases(self, degrader):
        result = degrader.inject_nulls(pct=0.15, inplace=True)
        assert result.isnull().sum().sum() > 0

    def test_null_rate_approximate(self, clean_df):
        """Null rate across all columns should be close to requested pct."""
        d = DQDegrader(clean_df, random_seed=42)
        result = d.inject_nulls(pct=0.20, inplace=True)
        actual_null_rate = result.isnull().sum().sum() / result.size
        # Allow ±5% tolerance
        assert abs(actual_null_rate - 0.20) < 0.05

    def test_targeted_columns_only(self, degrader):
        """Nulls should only appear in specified columns."""
        result = degrader.inject_nulls(pct=0.20, columns=["age"], inplace=True)
        assert result["age"].isnull().sum() > 0
        assert result["balance"].isnull().sum() == 0
        assert result["job"].isnull().sum() == 0

    def test_shape_unchanged(self, degrader, clean_df):
        result = degrader.inject_nulls(pct=0.10, inplace=True)
        assert result.shape == clean_df.shape

    def test_reproducibility(self, clean_df):
        """Two degraders with the same seed should produce identical null masks."""
        d1 = DQDegrader(clean_df, random_seed=42)
        d2 = DQDegrader(clean_df, random_seed=42)
        r1 = d1.inject_nulls(pct=0.15, inplace=True)
        r2 = d2.inject_nulls(pct=0.15, inplace=True)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_differ(self, clean_df):
        """Two degraders with different seeds should produce different null masks."""
        d1 = DQDegrader(clean_df, random_seed=1)
        d2 = DQDegrader(clean_df, random_seed=99)
        r1 = d1.inject_nulls(pct=0.20, inplace=True)
        r2 = d2.inject_nulls(pct=0.20, inplace=True)
        # At least some null positions should differ
        assert not r1.isnull().equals(r2.isnull())

    def test_raises_on_invalid_pct_zero(self, degrader):
        with pytest.raises(ValueError, match="pct"):
            degrader.inject_nulls(pct=0.0, inplace=True)

    def test_raises_on_invalid_pct_one(self, degrader):
        with pytest.raises(ValueError, match="pct"):
            degrader.inject_nulls(pct=1.0, inplace=True)

    def test_raises_on_missing_column(self, degrader):
        with pytest.raises(ValueError, match="not found"):
            degrader.inject_nulls(pct=0.10, columns=["nonexistent_col"], inplace=True)

    def test_null_rate_property_updates(self, degrader):
        before = degrader.null_rate
        degrader.inject_nulls(pct=0.20)
        after = degrader.null_rate
        assert after > before


# ------------------------------------------------------------------
# INJECT LABEL NOISE
# ------------------------------------------------------------------

class TestInjectLabelNoise:

    def test_some_labels_flipped(self, degrader, clean_df):
        result = degrader.inject_label_noise(target_col="y", pct=0.10, inplace=True)
        # At least some labels should differ from the original
        original_y = clean_df["y"].values
        result_y = result["y"].values
        n_changed = (original_y != result_y).sum()
        assert n_changed > 0

    def test_flip_count_approximate(self, clean_df):
        """Number of flipped labels should be close to pct * n_rows."""
        d = DQDegrader(clean_df, random_seed=42)
        result = d.inject_label_noise(target_col="y", pct=0.10, inplace=True)
        n_changed = (clean_df["y"].values != result["y"].values).sum()
        expected = int(len(clean_df) * 0.10)
        assert abs(n_changed - expected) <= 2  # Allow tiny rounding tolerance

    def test_only_target_column_affected(self, degrader, clean_df):
        result = degrader.inject_label_noise(target_col="y", pct=0.10, inplace=True)
        for col in ["age", "balance", "duration", "job", "marital"]:
            pd.testing.assert_series_equal(
                clean_df[col].reset_index(drop=True),
                result[col].reset_index(drop=True),
            )

    def test_label_values_remain_binary(self, degrader, clean_df):
        """After flipping, only original label values should appear."""
        original_labels = set(clean_df["y"].unique())
        result = degrader.inject_label_noise(target_col="y", pct=0.20, inplace=True)
        result_labels = set(result["y"].dropna().unique())
        assert result_labels == original_labels

    def test_reproducibility(self, clean_df):
        d1 = DQDegrader(clean_df, random_seed=42)
        d2 = DQDegrader(clean_df, random_seed=42)
        r1 = d1.inject_label_noise(target_col="y", pct=0.15, inplace=True)
        r2 = d2.inject_label_noise(target_col="y", pct=0.15, inplace=True)
        pd.testing.assert_series_equal(r1["y"], r2["y"])

    def test_raises_on_missing_target_col(self, degrader):
        with pytest.raises(ValueError, match="not found"):
            degrader.inject_label_noise(target_col="nonexistent", pct=0.10)

    def test_raises_on_non_binary_target(self, clean_df):
        """Should raise if target has more than 2 unique values."""
        df = clean_df.copy()
        df["multiclass"] = np.random.choice(["a", "b", "c"], size=len(df))
        d = DQDegrader(df, random_seed=42)
        with pytest.raises(ValueError, match="binary"):
            d.inject_label_noise(target_col="multiclass", pct=0.10)


# ------------------------------------------------------------------
# INJECT DUPLICATES
# ------------------------------------------------------------------

class TestInjectDuplicates:

    def test_row_count_increases(self, degrader, clean_df):
        result = degrader.inject_duplicates(pct=0.20, inplace=True)
        assert len(result) > len(clean_df)

    def test_row_count_approximate(self, clean_df):
        """New row count should be approximately original + pct * original."""
        d = DQDegrader(clean_df, random_seed=42)
        result = d.inject_duplicates(pct=0.25, inplace=True)
        expected = len(clean_df) + int(len(clean_df) * 0.25)
        assert abs(len(result) - expected) <= 2

    def test_column_count_unchanged(self, degrader, clean_df):
        result = degrader.inject_duplicates(pct=0.20, inplace=True)
        assert result.shape[1] == clean_df.shape[1]

    def test_duplicates_are_real_rows(self, degrader, clean_df):
        """All rows in the degraded DataFrame should exist in the original."""
        result = degrader.inject_duplicates(pct=0.30, inplace=True)
        # Every row in result should match at least one row in original
        merged = result.merge(clean_df, how="left", indicator=True)
        assert (merged["_merge"] == "both").all()

    def test_reproducibility(self, clean_df):
        d1 = DQDegrader(clean_df, random_seed=42)
        d2 = DQDegrader(clean_df, random_seed=42)
        r1 = d1.inject_duplicates(pct=0.20, inplace=True)
        r2 = d2.inject_duplicates(pct=0.20, inplace=True)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_raises_on_invalid_pct(self, degrader):
        with pytest.raises(ValueError, match="pct"):
            degrader.inject_duplicates(pct=1.5)


# ------------------------------------------------------------------
# INJECT OUTLIERS
# ------------------------------------------------------------------

class TestInjectOutliers:

    def test_numeric_columns_affected(self, degrader, clean_df):
        """Numeric columns should have changed values after outlier injection."""
        result = degrader.inject_outliers(
            columns=["age", "balance"], sigma=3.0, pct=0.10, inplace=True
        )
        changed_age = (clean_df["age"].values != result["age"].values).sum()
        changed_balance = (clean_df["balance"].values != result["balance"].values).sum()
        assert changed_age > 0
        assert changed_balance > 0

    def test_categorical_columns_unaffected(self, degrader, clean_df):
        """Categorical columns must not be modified by outlier injection."""
        result = degrader.inject_outliers(sigma=3.0, pct=0.10, inplace=True)
        for col in ["job", "marital", "y"]:
            pd.testing.assert_series_equal(
                clean_df[col].reset_index(drop=True),
                result[col].reset_index(drop=True),
            )

    def test_defaults_to_all_numeric(self, clean_df):
        """When columns=None, all numeric columns should be targeted."""
        d = DQDegrader(clean_df, random_seed=42)
        result = d.inject_outliers(sigma=3.0, pct=0.10, inplace=True)
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            changed = (clean_df[col].values != result[col].values).sum()
            assert changed > 0, f"Expected changes in numeric column '{col}'"

    def test_shape_unchanged(self, degrader, clean_df):
        result = degrader.inject_outliers(sigma=3.0, pct=0.10, inplace=True)
        assert result.shape == clean_df.shape

    def test_reproducibility(self, clean_df):
        d1 = DQDegrader(clean_df, random_seed=42)
        d2 = DQDegrader(clean_df, random_seed=42)
        r1 = d1.inject_outliers(sigma=3.0, pct=0.10, inplace=True)
        r2 = d2.inject_outliers(sigma=3.0, pct=0.10, inplace=True)
        pd.testing.assert_frame_equal(r1, r2)

    def test_raises_when_no_numeric_cols(self):
        """Should raise if DataFrame has no numeric columns."""
        df_cat = pd.DataFrame(
            {"a": ["x", "y"] * 50, "b": ["p", "q"] * 50}
        )
        d = DQDegrader(df_cat, random_seed=42)
        with pytest.raises(ValueError, match="numeric"):
            d.inject_outliers(sigma=3.0, pct=0.10)

    def test_raises_on_invalid_pct(self, degrader):
        with pytest.raises(ValueError, match="pct"):
            degrader.inject_outliers(sigma=3.0, pct=0.0)


# ------------------------------------------------------------------
# PIPELINE COMPOSER
# ------------------------------------------------------------------

class TestPipelineComposer:

    def test_chaining_returns_self(self, degrader):
        result = degrader.inject_nulls(pct=0.10)
        assert isinstance(result, DQDegrader)

    def test_result_returns_dataframe(self, degrader):
        df = degrader.inject_nulls(pct=0.10).result()
        assert isinstance(df, pd.DataFrame)

    def test_full_pipeline_applies_all_steps(self, clean_df):
        result = (
            DQDegrader(clean_df, random_seed=42)
            .inject_nulls(pct=0.10)
            .inject_label_noise(target_col="y", pct=0.05)
            .inject_duplicates(pct=0.15)
            .inject_outliers(sigma=3.0, pct=0.05)
            .result()
        )
        # Nulls injected
        assert result.isnull().sum().sum() > 0
        # Rows increased due to duplicates
        assert len(result) > len(clean_df)

    def test_pipeline_summary_records_steps(self, clean_df):
        d = (
            DQDegrader(clean_df, random_seed=42)
            .inject_nulls(pct=0.10)
            .inject_label_noise(target_col="y", pct=0.05)
        )
        summary = d.pipeline_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "operation" in summary.columns

    def test_reset_restores_original(self, clean_df):
        d = DQDegrader(clean_df, random_seed=42)
        d.inject_nulls(pct=0.30)
        d.reset()
        pd.testing.assert_frame_equal(d.result(), clean_df.reset_index(drop=True))

    def test_reset_clears_pipeline_log(self, clean_df):
        d = DQDegrader(clean_df, random_seed=42)
        d.inject_nulls(pct=0.10)
        d.inject_duplicates(pct=0.10)
        assert len(d.pipeline_summary()) == 2
        d.reset()
        assert len(d.pipeline_summary()) == 0

    def test_reset_reseeds_rng(self, clean_df):
        """After reset, the same operations should produce identical results."""
        d = DQDegrader(clean_df, random_seed=42)
        r1 = d.inject_nulls(pct=0.15).result()
        d.reset()
        r2 = d.inject_nulls(pct=0.15).result()
        pd.testing.assert_frame_equal(r1, r2)

    def test_result_is_copy(self, degrader):
        """Modifying result() output should not affect internal working DataFrame."""
        result = degrader.inject_nulls(pct=0.10).result()
        result["age"] = -999
        assert (degrader._working["age"] != -999).all()


# ------------------------------------------------------------------
# PROPERTIES
# ------------------------------------------------------------------

class TestProperties:

    def test_null_rate_zero_on_clean(self, degrader):
        assert degrader.null_rate == 0.0

    def test_null_rate_nonzero_after_injection(self, degrader):
        degrader.inject_nulls(pct=0.20)
        assert degrader.null_rate > 0.0

    def test_shape_reflects_duplicates(self, degrader, clean_df):
        degrader.inject_duplicates(pct=0.20)
        assert degrader.shape[0] > clean_df.shape[0]

    def test_original_unchanged_after_pipeline(self, clean_df):
        d = DQDegrader(clean_df, random_seed=42)
        d.inject_nulls(pct=0.30).inject_duplicates(pct=0.50)
        pd.testing.assert_frame_equal(
            d.original.reset_index(drop=True),
            clean_df.reset_index(drop=True),
        )
