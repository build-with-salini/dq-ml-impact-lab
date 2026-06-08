# 🏦 dq-ml-impact-lab

> *"Measuring how data quality degradation affects machine learning performance — applied to banking marketing data."*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Dataset: UCI](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing-lightgrey?style=flat-square)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## 📌 What Is This?

Poor data quality silently kills machine learning models. A model trained on dirty data does not just underperform — it **learns the wrong patterns entirely**.

This project provides a reproducible, end-to-end laboratory for:

1. **Profiling data quality** using ML-powered anomaly detection
2. **Simulating controlled DQ degradation** (nulls, label noise, duplicates, outliers)
3. **Measuring the downstream impact** on ML model performance

Applied to the **UCI Bank Marketing dataset**, which simulates real-world customer contact data from a Portuguese banking institution — a context directly relevant to fintech, digital banking, and CRM analytics.

> **Reusability note:** The profiler and degradation engine are dataset-agnostic. Swap the dataset, reuse the framework.

---

## 🎯 Who Is This For?

| Role | What You Will Find Here |
|---|---|
| **Data Engineers** | Modular `src/` pipeline, degradation engine, DQ scoring logic |
| **Data Scientists** | ML impact experiments, model performance curves, notebook walkthroughs |
| **Data Governance / Stewards** | DQ scorecard framework, dimension-level scoring, governance narrative |

---

## 📂 Repository Structure

```
dq-ml-impact-lab/
│
├── data/
│   ├── raw/                        # Original UCI dataset (not tracked in git)
│   └── degraded/                   # Programmatically degraded dataset versions
│
├── notebooks/
│   ├── 01_eda_and_profiling.ipynb  # EDA + baseline DQ metrics
│   ├── 02_dq_profiler.ipynb        # ML-powered DQ scoring per feature
│   ├── 03_degradation_experiments.ipynb  # Controlled DQ injection
│   └── 04_ml_impact_analysis.ipynb # Model performance vs DQ degradation
│
├── src/
│   ├── dq_profiler.py              # ML-powered DQ scoring module
│   ├── degrader.py                 # Controlled DQ degradation functions
│   ├── trainer.py                  # Model training + evaluation wrapper
│   └── utils.py                    # Shared helpers
│
├── app/
│   └── streamlit_app.py            # Streamlit dashboard entry point
│
├── tests/
│   └── test_degrader.py            # Unit tests for degradation logic
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**UCI Bank Marketing Dataset**
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Citation:** Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
- **Size:** ~45,000 rows, 17 features
- **Task:** Predict whether a client will subscribe to a term deposit (binary classification)
- **DQ Characteristics:** Contains `unknown` coded missings, class imbalance, and mixed feature types — realistic DQ challenges

> ⚠️ No proprietary or sensitive data is used in this project. The UCI dataset is fully public.

---

## 🔬 What The Experiments Do

### DQ Profiling (`02_dq_profiler.ipynb`)
- Applies **Isolation Forest** anomaly detection per numeric column
- Computes per-feature DQ scores across dimensions: completeness, consistency, anomaly rate
- Outputs a **DQ Scorecard** — a DataFrame ranking features by quality

### Degradation Engine (`03_degradation_experiments.ipynb`)
Programmatically injects controlled DQ issues:

| Degradation Type | Levels Applied |
|---|---|
| Null injection | 5%, 15%, 30% of values |
| Label noise | 5%, 10%, 20% target flips |
| Duplicate rows | 10%, 25%, 50% inflation |
| Outlier injection | Gaussian noise at varying sigma |

### ML Impact Analysis (`04_ml_impact_analysis.ipynb`)
- Trains **Logistic Regression** and **Random Forest** classifiers on each degraded version
- Measures Accuracy and F1-score at each degradation level
- Produces the **hero visualisation**: DQ degradation level vs model performance curve

---

## 🖥️ Streamlit Dashboard

The dashboard makes the experiments interactive and accessible without running notebooks.

| Page | Description |
|---|---|
| 🏠 **Overview** | Project summary and dataset snapshot |
| 🔍 **DQ Profiler** | Upload any CSV → receive a DQ scorecard |
| ⚗️ **Degradation Lab** | Sliders to control degradation → live DQ score update |
| 📉 **ML Impact** | Interactive chart: DQ score vs model accuracy |
| 📋 **Report** | Downloadable DQ summary as CSV |

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/dq-ml-impact-lab.git
cd dq-ml-impact-lab
pip install -r requirements.txt
```

### Download the Dataset

```bash
# Option 1: Manual download
# Visit https://archive.ics.uci.edu/dataset/222/bank+marketing
# Place bank-additional-full.csv in data/raw/

# Option 2: Via ucimlrepo (if installed)
python -c "from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=222); d.data.features.to_csv('data/raw/bank_features.csv')"
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

---

## 🧪 Running Tests

```bash
pytest tests/test_degrader.py -v
```

Tests cover the core degradation logic to ensure injected DQ issues are reproducible and correctly scoped.

---

## 🔑 Key Findings

> *(To be updated after experiments are complete)*

Preliminary observations:
- Null injection at **30%** is expected to cause measurable F1 decline in both classifiers
- Label noise is hypothesised to impact **Logistic Regression more severely** than Random Forest
- The DQ Scorecard reveals feature-level quality variance even in the clean dataset

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data profiling | `ydata-profiling`, `pandas`, `scipy` |
| ML anomaly detection | `scikit-learn` (Isolation Forest) |
| ML classification | `scikit-learn` (Logistic Regression, Random Forest) |
| Dashboard | `Streamlit` |
| Notebook environment | `Jupyter` |
| Testing | `pytest` |

---

## ♻️ Reusing This Framework

The `src/` modules are designed to be dataset-agnostic:

```python
from src.dq_profiler import DQProfiler
from src.degrader import DQDegrader

# Works on any pandas DataFrame
profiler = DQProfiler(df)
scorecard = profiler.score()

degrader = DQDegrader(df)
degraded_df = degrader.inject_nulls(pct=0.15)
```

Swap the UCI dataset for your own tabular data and the pipeline applies directly.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Salini Anbalagan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/salini-anbalagan/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/build-with-salini)

---
*Built as part of an open data portfolio initiative. Feedback and contributions welcome.*
