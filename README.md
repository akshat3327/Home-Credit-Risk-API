# 🏦 Explainable Credit Risk Decision Engine

🚀 **Live API & Swagger UI:** https://home-credit-risk-api-1.onrender.com/docs

## 📌 System Overview
End-to-end Machine Learning microservice designed to assess loan default risk. Moving beyond standard classification, this system implements a **hybrid soft-voting ensemble (50% XGBoost, 50% CatBoost)** wrapped in a FastAPI backend. The architecture prioritizes strict schema validation, inference consistency, algorithmic explainability, and low-latency execution,combining machine learning with a production-ready FastAPI backend

## 🏗️ Inference Architecture & Artifact Management

To ensure identical preprocessing during both training and inference, the deployment pipeline relies on serialized artifacts loaded at runtime.

- **`preprocessor.pkl`**: Applies the same imputation, scaling, and ordinal encoding used during model training.
- **`features.pkl`**: Preserves feature ordering between training and inference, preventing feature-order mismatches during prediction.
- **Dynamic Feature Generation**: Financial ratios and interaction features are generated from incoming request data before inference.

```text
[Client Request] 
       │
       ▼
[FastAPI Router] ➔ [Pydantic Strict Validation] 
       │
       ▼
[preprocessor.pkl: Identical Train/Inference Transformation] 
       │
       ▼
[Dynamic Feature Engineering (Financial Ratios & Interactions)] 
       │
       ▼
[features.pkl: Strict Feature Alignment & Ordering]
       │
       ▼
[XGBoost (50%) + CatBoost (50%) Soft-Voting Ensemble] 
       │
       ▼
[SHAP Explainer (Calculated per API Request)] 
       │
       ▼
[JSON Response (Risk Probability, Classification, SHAP Values)]
```
## 📊 Model Performance & Ensemble Logic

The deployed model uses a 50:50 soft-voting ensemble of XGBoost and CatBoost. Combining both models produced more consistent predictions than using either model individually.

- **Original pipeline:** ~0.78 ROC-AUC using Application, Bureau, and Previous Application tables.
- **Deployed pipeline:** ~0.77 ROC-AUC using only Application-level features.

**Key Finding (`EXT_SOURCE_MEAN`)**: During experimentation, combining the three external risk scores into a single mean feature consistently ranked among the most important features in both feature importance and SHAP analysis, improving overall model performance.
Engineering Trade-Offs
The decision to drop the multi-table architecture in favor of the Application-only pipeline was a deliberate production choice.

The Trade-off: Sacrificed a negligible ~0.01 ROC-AUC.

The Engineering ROI: Reduced API payload complexity by >90%, eliminated the need for complex client-side relational data joins, minimized request latency, and drastically simplified deployment.This trade-off favors a simpler, faster deployment over a small improvement in ROC-AUC
🔍 Explainability (XAI) & Financial Compliance
Designed to support model transparency and financial auditing use cases through SHAP explanations, this system does not operate as a black box.
SHAP (SHapley Additive exPlanations) values are computed dynamically for every single API request. The JSON response explicitly returns the top contributing features and their specific impact values for that exact user, providing complete transparency into why an applicant was classified as low or high default risk. Furthermore, the API accepts a configurable threshold parameter, allowing financial institutions to dynamically adjust risk appetite without redeploying the underlying model.
📂 Production Project Structure
```text
loan-risk-platform/

│
├── api/                    # FastAPI endpoints, routers, and strict Pydantic schemas
├── data/                   # Raw & processed datasets (Ignored in Git via .gitignore)
├── models/                 # Serialized pipeline artifacts protecting inference consistency
│   ├── features.pkl        # Enforces final feature ordering 
│   └── saved_models/
│       ├── xgb_model.pkl   # Serialized XGBoost estimator
│       ├── cat_model.pkl   # Serialized CatBoost estimator
│       └── preprocessor.pkl# Scikit-learn preprocessing pipeline
├── notebooks/              # Jupyter environment for EDA, feature selection, and validation
├── src/                    # Modularized Core ML Pipeline
│   ├── feature_engineering.py  # Logic for financial ratios (Credit-to-Income, etc.)
│   ├── explainability/         # SHAP TreeExplainer integration
│   ├── models/                 # Ensemble logic and prediction formatting
│   └── data_pipeline/          # Imputation and encoding scripts
├── tests/                  # Unit testing for API routes and ML pipeline
├── pyproject.toml          # Poetry dependency management for deterministic builds
└── poetry.lock             # Locked package versions
```
## 💻 Local Setup & Installation

This project utilizes **Poetry** for isolated, reproducible dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/akshat3327/Home-Credit-Risk-API.git
cd Home-Credit-Risk-API
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Run the API

```bash
poetry run uvicorn api.main:app --reload
```

### 4. Open Swagger UI

```
http://127.0.0.1:8000/docs
```
