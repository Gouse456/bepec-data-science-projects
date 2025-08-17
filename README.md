# bepec-data-science-projects
📌 Repository Description

This repository contains 4 real-world Data Science & Generative AI projects developed during the Bepec program. Each project demonstrates strong EDA, statistical analysis, ML modeling, optimization, and deployment practices across diverse domains including finance, banking, NLP, and recruitment analytics.

🔹 Projects Included

1.Credit Card Default Prediction

Processed 30k customer records with hypothesis testing (t-test, chi-square, Pearson) to identify default risk drivers such as credit limit, age, and gender.

Applied advanced resampling techniques (SMOTE, SMOTEENN, SMOTETomek, ADASYN) and built a meta-ensemble boosting model with threshold tuning.

Achieved ROC-AUC: 0.78, with defaulter recall: 0.65 and precision: 0.43 → enabling more reliable credit risk scoring.

2.Next Best Action Model – Standard Bank

Developed a churn classification engine using CatBoost + SMOTEENN (ROC-AUC: 0.86, F1: 0.82).

Built a recommendation engine for targeted retention strategies across 8 churn types.

Benchmarked multiple models (Random Forest, GridSearchCV), achieving macro-F1: 0.52 with strong performance on priority churn cases.

3.Product-Type Classification (NLP Classifier)

Designed an end-to-end multi-class text classifier for 15 product categories using 15k samples.

Built and optimized a TF-IDF + ML pipeline with probability matrices for per-class performance analysis.

Achieved macro-F1: 0.52 (accuracy: 0.43), with top performance in Tools Product Type (F1: 0.72).

4.Hiring Predictor Pipeline (End-to-End ML Pipeline)

Created a robust ML pipeline using ColumnTransformer + Pipeline for cleaning, encoding, scaling, and prediction.

Compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting; finalized Logistic Regression (ROC-AUC: 0.89).

Deployed via FastAPI & Streamlit with support for single candidate input and batch scoring through CSV uploads.

🚀 Key Highlights

Covers EDA, hypothesis testing, model selection, hyperparameter tuning, and deployment.

Implements advanced resampling, boosting, and ensemble strategies for imbalanced data.

Demonstrates production-ready deployment using FastAPI & Streamlit.

Provides business-relevant insights in finance, banking, NLP, and hiring domains.

👉 This repo reflects my ability to build scalable, end-to-end AI solutions — from data exploration to deployment — with a balance of technical rigor and practical business impact.
