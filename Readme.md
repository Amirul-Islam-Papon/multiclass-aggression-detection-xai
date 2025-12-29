# Multiclass Aggression Detection using Machine Learning & Explainable AI

## Overview
This project implements a multiclass NLP-based aggression detection system using supervised machine learning and Explainable AI (XAI) techniques. The goal is to classify text into multiple aggression categories while maintaining transparency in model decisions.

## Dataset
- Social media text samples(HateXplain, ucberkeley-dlab/measuring-hate-speech, Davidson HateSpeech )
- Preprocessed through noise removal, normalization, and feature engineering
- Handled class imbalance using resampling and class weighting

## Methodology
1. Data Cleaning & Exploratory Data Analysis (EDA)
2. Text Preprocessing & Feature Engineering
3. Multiclass Model Training (Random Forest)
4. Model Evaluation (Accuracy, Precision, Recall, F1-score)
5. Explainable AI using SHAP and LIME

## Results
- Accuracy: ~91%
- F1-score: ~0.89
- Improved minority class recall using balanced training strategies

## Explainability
SHAP and LIME were applied to interpret predictions and identify key linguistic features influencing aggression classification.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP, LIME


