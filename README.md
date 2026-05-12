# Fraud Detection in Financial Transactions

A complete machine learning project to catch fraudulent credit card transactions. Built with Python, scikit‑learn, and a real‑world dataset.

## The problem

A fintech company loses money to fraud every day. They need a model that flags suspicious transactions automatically – but they can’t afford to block too many legitimate customers.

**Goal:** Build a classifier that catches most fraud while keeping false alarms low enough for a fraud team to review.

## The data

- **Source:** Kaggle – Credit Card Fraud Detection  
- **284,807 transactions** | only **0.17%** are fraudulent  
- Features: PCA‑transformed (`V1` to `V28`), plus `Time` and `Amount`  
- No missing values, perfectly clean

## What I did

1. **Stratified split** – kept the same fraud percentage in train and test (0.17%)
2. **Handled imbalance** – used `class_weight='balanced'` in both Logistic Regression and Random Forest
3. **Trained two models** – compared interpretability (logistic) vs. performance (random forest)
4. **Tuned the decision threshold** – to reach a business‑friendly trade‑off:  
   *precision ≥ 70%* while maximising recall of fraud
5. **Evaluated with precision, recall, and ROC‑AUC** – not misleading accuracy

## Results

| Model | Precision (fraud) | Recall (fraud) | ROC‑AUC |
|-------|------------------|----------------|---------|
| Logistic Regression | 0.06 | 0.92 | 0.972 |
| Random Forest (default) | **0.96** | 0.74 | 0.953 |
| Random Forest (threshold = 0.08) | **0.71** | **0.89** | – |

**Business takeaway:**  
The tuned random forest catches **89% of fraud** while sending only **30% false alarms**. That means a small fraud team can handle the alerts without overwhelming customers.

## How to run it

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (recommended) or Jupyter.
2. Run all cells from top to bottom.
3. The dataset is loaded directly from a reliable mirror – no manual download needed.

**Requirements:**  
Python 3.9+, pandas, numpy, scikit‑learn, matplotlib, seaborn.  
(Colab has them all pre‑installed.)

## Files

- `fraud_detection.ipynb` – the complete notebook with markdown explanations
- `README.md` – this file

## What this proves

-  End‑to‑end data science pipeline (data → model → business insight)  
-  Handling severe class imbalance  
-  Model comparison and threshold tuning  
-  Writing for a non‑technical audience (the “business recommendation”)

## Connect

This project is part of my portfolio for data science roles.  
If you’d like to discuss the approach or see a version with on‑chain crypto data, send me a message.

---

**License:** MIT – feel free to reuse and adapt.
