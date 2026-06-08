"""
trainer.py
----------
Model Training and Evaluation Wrapper for dq-ml-impact-lab.

Provides a clean interface for:
    - Training Logistic Regression and Random Forest classifiers
    - Evaluating model performance (Accuracy, F1, Precision, Recall, ROC-AUC)
    - Running repeated experiments across degraded dataset versions
    - Returning structured results for downstream visualisation

Designed to be dataset-agnostic. Works on any binary classification DataFrame.

Usage:
    from src.trainer import ModelTrainer

    trainer = ModelTrainer(
        df=df,
        target_col="y",
        random_seed=42
    )

    # Train and evaluate a single model
    results = trainer.train_evaluate(model_type="random_forest")

    # Run full degradation experiment
    experiment_results = trainer.run_degradation_experiment(
        degraded_datasets=degraded_versions,  # dict of {label: df}
        model_type="both"
    )

Author: Salini Anbalagan
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Supported model types
ModelType = Literal["logistic_regression", "random_forest", "both"]

# Metrics computed for every evaluation
METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]


class ModelTrainer:
    """
    Model Training and Evaluation Wrapper.

    Handles preprocessing, training, and evaluation of binary classifiers
    on potentially degraded DataFrames.

    Preprocessing applied automatically:
        - Categorical columns    : Label encoding
        - Null values            : Median imputation (numeric), mode imputation (categorical)
        - Feature scaling        : StandardScaler (Logistic Regression only)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target column.
    target_col : str
        Name of the binary target column.
    random_seed : int, optional
        Seed for all random operations. Default is 42.
    cv_folds : int, optional
        Number of stratified cross-validation folds. Default is 5.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        random_seed: int = 42,
        cv_folds: int = 5,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        if not (2 <= cv_folds <= 20):
            raise ValueError("cv_folds must be between 2 and 20.")

        self._df = df.copy()
        self.target_col = target_col
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self._results_log: List[Dict] = []

        logger.info(
            f"ModelTrainer initialised | rows={len(df)} | target='{target_col}' | "
            f"cv_folds={cv_folds} | seed={random_seed}"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def train_evaluate(
        self,
        model_type: ModelType = "both",
        df: Optional[pd.DataFrame] = None,
        label: str = "baseline",
    ) -> pd.DataFrame:
        """
        Train and evaluate classifier(s) using stratified cross-validation.

        Parameters
        ----------
        model_type : str
            One of "logistic_regression", "random_forest", or "both".
        df : pd.DataFrame, optional
            DataFrame to use. Defaults to the initialisation DataFrame.
        label : str, optional
            Descriptive label for this run (e.g., "baseline", "nulls_15pct").
            Used in output DataFrame and experiment logs.

        Returns
        -------
        pd.DataFrame
            Results with columns: label, model, accuracy, f1, precision,
            recall, roc_auc, and standard deviations for each metric.

        Example
        -------
        >>> trainer.train_evaluate(model_type="random_forest", label="baseline")
        """
        eval_df = df if df is not None else self._df
        X, y = self._preprocess(eval_df)

        models = self._resolve_models(model_type)
        records = []

        for name, pipeline in models:
            logger.info(f"Training | model={name} | label={label} | folds={self.cv_folds}")
            cv_result = self._cross_validate(pipeline, X, y)
            row = {"label": label, "model": name}
            row.update(cv_result)
            records.append(row)
            self._results_log.append(row)

        return pd.DataFrame(records)

    def run_degradation_experiment(
        self,
        degraded_datasets: Dict[str, pd.DataFrame],
        model_type: ModelType = "both",
        include_baseline: bool = True,
    ) -> pd.DataFrame:
        """
        Run train_evaluate across multiple degraded dataset versions.

        Parameters
        ----------
        degraded_datasets : dict of {str: pd.DataFrame}
            Mapping of degradation label to degraded DataFrame.
            Example: {"nulls_5pct": df1, "nulls_15pct": df2, "nulls_30pct": df3}
        model_type : str
            One of "logistic_regression", "random_forest", or "both".
        include_baseline : bool
            If True, evaluates on the original initialisation DataFrame first.
            Default is True.

        Returns
        -------
        pd.DataFrame
            Stacked results across all degradation levels and models.
            Sorted by model then label for easy plotting.

        Example
        -------
        >>> results = trainer.run_degradation_experiment(
        ...     degraded_datasets={"nulls_15pct": df_nulls, "noise_10pct": df_noise},
        ...     model_type="both"
        ... )
        """
        all_results = []

        if include_baseline:
            logger.info("Evaluating baseline (clean dataset)...")
            baseline = self.train_evaluate(
                model_type=model_type,
                df=self._df,
                label="baseline",
            )
            all_results.append(baseline)

        for label, degraded_df in degraded_datasets.items():
            logger.info(f"Evaluating degraded version | label={label}")
            result = self.train_evaluate(
                model_type=model_type,
                df=degraded_df,
                label=label,
            )
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(["model", "label"]).reset_index(drop=True)

        logger.info(
            f"Degradation experiment complete | "
            f"versions={len(degraded_datasets) + int(include_baseline)} | "
            f"total_rows={len(combined)}"
        )
        return combined

    def results_log(self) -> pd.DataFrame:
        """
        Return a DataFrame of all train_evaluate calls made in this session.

        Returns
        -------
        pd.DataFrame
            Full log of all evaluation runs with metrics.

        Example
        -------
        >>> trainer.results_log()
        """
        if not self._results_log:
            logger.warning("No evaluations run yet. Call train_evaluate() first.")
            return pd.DataFrame()
        return pd.DataFrame(self._results_log)

    def baseline_scores(self, model_type: ModelType = "both") -> pd.DataFrame:
        """
        Convenience method: evaluate on the clean initialisation DataFrame.

        Parameters
        ----------
        model_type : str
            One of "logistic_regression", "random_forest", or "both".

        Returns
        -------
        pd.DataFrame
            Baseline evaluation results.

        Example
        -------
        >>> trainer.baseline_scores(model_type="random_forest")
        """
        return self.train_evaluate(model_type=model_type, label="baseline")

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.

        Steps:
            1. Separate features (X) from target (y)
            2. Encode target if non-numeric
            3. Label-encode categorical feature columns
            4. Return X as DataFrame, y as Series

        Note: Null imputation and scaling are handled inside the
        sklearn Pipeline per model (see _build_pipeline).
        """
        df = df.copy()

        # Separate target
        y_raw = df[self.target_col].copy()
        X = df.drop(columns=[self.target_col])

        # Encode target if needed
        if y_raw.dtype == object or str(y_raw.dtype) == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), name=self.target_col)
            logger.info(f"Target encoded | classes={list(le.classes_)}")
        else:
            y = y_raw.reset_index(drop=True)

        # Encode categorical features
        X = self._encode_categoricals(X)
        X = X.reset_index(drop=True)

        return X, y

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encode all object/category columns in X.
        Unknown/NaN values are encoded as -1 to preserve null signal.
        """
        X = X.copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            non_null = X[col].dropna().astype(str)
            le.fit(non_null)
            X[col] = X[col].apply(
                lambda v: le.transform([str(v)])[0]
                if pd.notna(v) and str(v) in le.classes_
                else -1
            )
        return X

    # ------------------------------------------------------------------
    # MODEL BUILDING
    # ------------------------------------------------------------------

    def _resolve_models(
        self, model_type: ModelType
    ) -> List[Tuple[str, Pipeline]]:
        """
        Resolve model type string to a list of (name, pipeline) tuples.
        """
        if model_type == "logistic_regression":
            return [("Logistic Regression", self._build_pipeline("lr"))]
        elif model_type == "random_forest":
            return [("Random Forest", self._build_pipeline("rf"))]
        elif model_type == "both":
            return [
                ("Logistic Regression", self._build_pipeline("lr")),
                ("Random Forest", self._build_pipeline("rf")),
            ]
        else:
            raise ValueError(
                f"model_type must be 'logistic_regression', 'random_forest', or 'both'. Got: {model_type}"
            )

    def _build_pipeline(self, model_key: str) -> Pipeline:
        """
        Build a sklearn Pipeline with imputation, optional scaling, and classifier.

        Logistic Regression pipeline: impute → scale → classify
        Random Forest pipeline:       impute → classify (trees are scale-invariant)
        """
        imputer = SimpleImputer(strategy="median")

        if model_key == "lr":
            return Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=self.random_seed,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
        elif model_key == "rf":
            return Pipeline(
                steps=[
                    ("imputer", imputer),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=100,
                            random_state=self.random_seed,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

    # ------------------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------------------

    def _cross_validate(
        self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Run stratified k-fold cross-validation and return mean + std per metric.

        Returns a flat dict:
            {
                "accuracy": 0.87, "accuracy_std": 0.02,
                "f1": 0.83,       "f1_std": 0.03,
                ...
            }
        """
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_seed,
        )

        scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        }

        try:
            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                error_score="raise",
            )
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            # Return NaN scores rather than crashing the experiment
            return {
                **{m: np.nan for m in METRICS},
                **{f"{m}_std": np.nan for m in METRICS},
            }

        result = {}
        for metric in METRICS:
            scores = cv_results[f"test_{metric}"]
            result[metric] = round(float(np.mean(scores)), 4)
            result[f"{metric}_std"] = round(float(np.std(scores)), 4)

        return result

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def n_experiments(self) -> int:
        """Return the number of train_evaluate calls made in this session."""
        return len(self._results_log)

    @property
    def target_distribution(self) -> pd.Series:
        """Return the value counts of the target column in the initialisation DataFrame."""
        return self._df[self.target_col].value_counts()

    def __repr__(self) -> str:
        return (
            f"ModelTrainer("
            f"rows={len(self._df)}, "
            f"target='{self.target_col}', "
            f"cv_folds={self.cv_folds}, "
            f"experiments_run={self.n_experiments}, "
            f"seed={self.random_seed})"
        )
