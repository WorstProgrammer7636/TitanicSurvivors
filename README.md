 
# Titanic Survival Prediction (Kaggle)

This project explores binary classification techniques using the Titanic dataset from Kaggle. It aims to predict passenger survival using classical and modern machine learning approaches while emphasizing clean pipelines, interpretability, and progressive model tuning.

## Features

- Feature engineering (Title extraction, Family size, Deck from cabin)
- One-hot encoding, imputation, scaling
- Logistic Regression with cross-validation
- Random Forest classifier (benchmarking)
- Clean separation between preprocessing, modeling, and main pipeline

## Planned Enhancements

- Add XGBoost, LightGBM models
- Model interpretation (SHAP values, feature importances)
- Ensemble modeling
- Threshold tuning and ROC optimization
- Cross-validation on multiple seeds

## Project Structure

- `main.py` - Runs end-to-end pipeline
- `feature_engineering.py` - Title/Deck/FamilySize logic
- `preprocessing.py` - Scaling, encoding, imputation
- `modeling.py` - Model training functions
- `data/` - Contains raw and processed CSVs

## Goal

Build an end-to-end ML pipeline that’s modular, extendable, and readable and improve upon the baseline model with advanced feature engineering and ensemble methods.
