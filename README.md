# Workers' Compensation Claim Classification

![Grade](https://img.shields.io/badge/Grade-20%2F20-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

Multiclass classification model to predict the **Claim Injury Type** awarded by the New York Workers' Compensation Board (WCB), developed as part of the Machine Learning course at NOVA IMS (2024/2025).

---

## Problem

The WCB receives thousands of workplace injury claims and must decide what type of compensation benefit to award. The goal is to automate this decision using historical claim data from 2020–2022, predicting one of **8 injury type classes** for new claims.

---

## Approach

### 1. Data Exploration & Preprocessing — [`01_data_exploration_preprocessing.ipynb`](./01_data_exploration_preprocessing.ipynb)

- Analysis of missing values, distributions, and class imbalance across **593,471 training records** and 33 features
- Date feature engineering: extracted gaps between key dates (accident, assembly, C-2, C-3); green/red flag columns for chronological consistency and regulatory compliance
- KNN imputation for missing values (with MinMax scaling applied before and reversed after), run on a 15% sample due to computational cost
- Categorical encoding: frequency encoding (Carrier Name, District, County), one-hot encoding (Carrier Type, Medical Fee Region), target encoding (WCIO and Industry descriptions)
- Domain-specific rules: C-2/C-3 date logic, ZIP code processing, wage outlier capping at IQR upper limit

### 2. Modelling & Ensemble — [`02_modelling_ensemble.ipynb`](./02_modelling_ensemble.ipynb)

- **Feature selection:** intersection of Mutual Information (threshold 0.02), LASSO, and RFE — reducing features roughly by half
- **Models tested:** Logistic Regression, Random Forest, XGBoost, LightGBM, MLP
- **Class imbalance:** SMOTE + RandomUnderSampler, jointly optimized with Optuna
- **Ensemble:** Weighted probability combination of all 5 models with Optuna-tuned weights across 3 Stratified K-Folds — Random Forest received weight 0 in all folds; XGBoost was consistently the dominant model
- **Explainability:** LIME for per-class feature contribution analysis

### 3. Open-Ended Extensions

- **Agreement Reached prediction:** separate model to predict this feature for the test set (F1 macro 0.71), used as a predictor in the main model — yielding a 1–1.5% improvement in F1 macro
- **Web app:** interactive tool where a WCB employee can input claim features and get a real-time Claim Injury Type prediction from a pre-trained, Optuna-tuned XGBoost model *(not included — dataset was private and provided solely for the course competition)*

---

## Results

| Model               | F1-Macro (avg. 3 folds) |
|---------------------|-------------------------|
| Logistic Regression | ~0.40                   |
| Random Forest       | ~0.44                   |
| MLP                 | ~0.445                  |
| LightGBM            | ~0.482                  |
| XGBoost             | ~0.493                  |
| **Weighted Ensemble** | **0.497**             |

**Kaggle leaderboard (best fold):** 0.439 — best score during experimentation: 0.515.

> The gap between cross-validation and Kaggle scores reflects some overfitting, identified as a key area for future work. The Kaggle competition was private (class-internal).

---

## Repository Structure

```
├── 01_data_exploration_preprocessing.ipynb   # Data exploration and preprocessing
├── 02_modelling_ensemble.ipynb               # Feature selection, modelling, ensemble
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The datasets (`train_data.csv`, `test_data.csv`) are not included as they were provided privately for the course competition.

---

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

1. `01_data_exploration_preprocessing.ipynb`
2. `02_modelling_ensemble.ipynb`

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `imbalanced-learn` · `Optuna` · `LIME` · `pandas` · `NumPy`

---

## Author

**Guilherme Godinho**
MSc Data Science and Advanced Analytics — NOVA IMS, Lisboa

[![LinkedIn](https://img.shields.io/badge/LinkedIn-guilherme--godinho-0077B5?logo=linkedin)](https://www.linkedin.com/in/guilherme-godinho-2493a6239/)
[![Email](https://img.shields.io/badge/Email-guilhermefloresgodinho%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:guilhermefloresgodinho@gmail.com)
