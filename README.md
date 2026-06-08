# Workers' Compensation Claim Classification

Multiclass classification model to predict the **Claim Injury Type** awarded by the New York Workers' Compensation Board (WCB), developed as part of the Machine Learning course at NOVA IMS (2024/2025).

**Grade: 20/20**

---

## Problem

The WCB receives thousands of workplace injury claims and must decide what type of compensation benefit to award. The goal is to automate this decision using historical claim data from 2020–2022, predicting one of 8 injury type classes for new claims.

## Approach

### 1. Data Exploration & Preprocessing (`ML_Project_Group50_DataExp_Prep.ipynb`)
- Analysis of missing values, distributions, and class imbalance across 593,471 training records and 33 features
- Date feature engineering: extracted gaps between key dates (accident, assembly, C-2, C-3); green/red flag columns for chronological consistency and regulatory compliance
- KNN imputation for missing values (with MinMax scaling applied before and reversed after), run on a 15% sample due to computational cost
- Categorical encoding: frequency encoding (Carrier Name, District, County), one-hot encoding (Carrier Type, Medical Fee Region), target encoding (WCIO and Industry descriptions)
- Domain-specific rules: C-2/C-3 date logic, ZIP code processing, wage outlier capping at IQR upper limit

### 2. Modelling (`ML_Project_Group50_Exodia.ipynb`)
- **Feature selection:** intersection of Mutual Information (threshold 0.02), LASSO, and RFE — reducing features roughly by half
- **Models tested:** Logistic Regression, Random Forest, XGBoost, LightGBM, MLP
- **Class imbalance:** SMOTE + RandomUnderSampler, jointly optimized with Optuna
- **Ensemble:** Weighted probability combination of all 5 models with Optuna-tuned weights across 3 Stratified K-Folds — notably, Random Forest received weight 0 in all folds and was never used in the final ensemble; XGBoost was consistently the dominant model
- **Explainability:** LIME for per-class feature contribution analysis

### 3. Open-Ended Extensions
- **Agreement Reached prediction:** separate model to predict this feature for the test set (F1 macro 0.71), which was then used as a predictor in the main model — yielding a 1–1.5% improvement in F1 macro
- **Web app:** interactive tool where a WCB employee can input claim features and get a real-time Claim Injury Type prediction from a pre-trained, Optuna-tuned XGBoost model

---

### Results

| Model | F1-Macro (resampled, avg 3 folds) |
|---|---|
| Logistic Regression | ~0.40 |
| Random Forest | ~0.44 |
| XGBoost | ~0.493 |
| LightGBM | ~0.482 |
| MLP | ~0.445 |
| **Weighted Ensemble** | **0.497** |

**Kaggle leaderboard (best fold):** 0.439 — best score during experimentation: 0.515.

> The gap between cross-validation and Kaggle scores suggests some overfitting, identified as a key area for future work.

> Kaggle competition was private (class-internal). Final project grade: **20/20**.

---

## Repository Structure

```
├── ML_Project_Group50_DataExp_Prep.ipynb   # Data exploration and preprocessing
├── ML_Project_Group50_Exodia.ipynb         # Feature selection, modelling, ensemble
├── requirements.txt
└── README.md
```

> **Note:** The datasets (`train_data.csv`, `test_data.csv`) are not included as they were provided privately for the course competition.

---

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in order:
1. `ML_Project_Group50_DataExp_Prep.ipynb`
2. `ML_Project_Group50_Exodia.ipynb`

---

## Tech Stack

Python · scikit-learn · XGBoost · LightGBM · imbalanced-learn · Optuna · LIME · pandas · NumPy
