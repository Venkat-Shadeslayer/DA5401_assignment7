# DA5401 — Assignment 7: Model Selection

This repository contains the Jupyter Notebook **`DA5401_A7.ipynb`**, which demonstrates the process of **model selection and evaluation** using the **Statlog (Landsat Satellite) Dataset** from the UCI Machine Learning Repository.
Name: Venkata Sai Vihswesvar SV
GITHUB ID: Venkat-Shadeslayer
Roll number: BE22B042
---

## 📘 Overview

This assignment explores different machine learning models to identify the best-performing classifier for a multiclass satellite image dataset. The workflow covers data preprocessing, training, evaluation, and visualization.

---

## 🧩 Contents

- **Dataset:** Statlog (Landsat Satellite) Dataset (UCI Repository)
- **Notebook:** notebooks/`assignment7_be22b042.ipynb`
- **Techniques Used:**
  - Data preprocessing and cleaning
  - Train-test split
  - Model training and hyperparameter tuning
  - Model evaluation and comparison
  - Visualization of ROC and Precision-Recall curves (macro-averaged, One-vs-Rest)

---

## ⚙️ Models Implemented

| Model | Library | Notes |
|--------|----------|--------|
| K-Nearest Neighbors (KNN) | `scikit-learn` | Distance-based classification |
| Decision Tree | `scikit-learn` | Tree-based model for interpretability |
| Logistic Regression | `scikit-learn` | Multinomial logistic regression |
| Gaussian Naive Bayes | `scikit-learn` | Probabilistic baseline |
| Support Vector Classifier (SVC) | `scikit-learn` | Kernel-based classification |
| Dummy Classifier | `scikit-learn` | Baseline reference |
| *(Optional)* XGBoost | `xgboost` | Gradient boosting (optional extension) |

---

## 📊 Evaluation Metrics

The notebook computes:

- **Accuracy**
- **Weighted F1-score**
- **Macro-averaged ROC curves (One-vs-Rest)**
- **Precision–Recall curves**

Each model is ranked by performance, with brief synthesis and analysis.

---

## 🧠 Key Takeaways

- Different classifiers exhibit varying performance trade-offs on multiclass problems.
- ROC and Precision–Recall curves provide deeper insights beyond accuracy.
- Ensemble and kernel-based methods tend to outperform simpler baselines on high-dimensional data.

---

## 🛠️ Requirements

Make sure the following packages are installed:

```bash
pip install numpy pandas scikit-learn matplotlib xgboost
