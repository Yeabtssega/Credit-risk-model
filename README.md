Credit Risk Model – Bati Bank
Author: Yeabtsega Tilahun
Organization: Bati Bank
Project: Credit Scoring for Buy-Now-Pay-Later Loan Approvals
Submission: Final | July 1, 2025

📘 Project Overview
Bati Bank is partnering with a successful eCommerce company to launch a Buy-Now-Pay-Later (BNPL) service. As the Analytics Engineer, you're tasked with designing and deploying a Credit Scoring Model that assesses the credit risk of customers using behavioral transaction data.

This project delivers a complete ML workflow—from data preprocessing and feature engineering to risk modeling, CI/CD deployment, and an interactive prediction API.

🎯 Business Objective
Credit scoring quantifies the likelihood that a borrower will default on a loan. Our key innovation is transforming customer transaction behavior (RFM metrics) into a proxy for credit risk, enabling risk-based loan decisioning.

✅ Goal: Assign risk probability scores and recommend credit terms for new customers using transaction behavior.

💼 Credit Scoring Business Understanding
1. Why does Basel II demand an interpretable model?
Basel II emphasizes transparency, accountability, and regulatory compliance. Financial institutions must explain how and why a credit decision was made. Therefore, interpretable models (like logistic regression with Weight of Evidence) are preferred for traceability.

2. Why create a proxy target variable?
The dataset lacks an explicit “default” label. We create a proxy using RFM clustering, defining "high-risk" customers based on low frequency and monetary values. This approach enables modeling, but comes with business risk: poor proxy definitions may misclassify customers, leading to biased decisions.

3. Simple vs Complex Models: What’s the trade-off?
Simple (e.g., Logistic Regression + WoE): Interpretable, compliant, easier to audit

Complex (e.g., Gradient Boosting): Higher accuracy but less transparent
In regulated sectors, the cost of opacity may outweigh marginal accuracy gains.

🗂️ Project Structure
bash
Copy
Edit
credit-risk-model/
├── .github/workflows/ci.yml        # CI/CD: lint + unit tests
├── data/
│   ├── raw/                        # Raw data (ignored in Git)
│   └── processed/                  # Processed, ready-to-train
├── notebooks/
│   └── 1.0-eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── data_processing.py          # Feature Engineering
│   ├── train.py                    # Model training & MLflow
│   ├── predict.py                  # Model inference
│   └── api/
│       ├── main.py                 # FastAPI backend
│       └── pydantic_models.py      # Input/Output schemas
├── tests/
│   └── test_data_processing.py     # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
📊 Key Tasks and Results
✅ Task 1 – Basel II & Business Framing
Defined credit risk modeling scope

Identified need for transparency and proxy labels

✅ Task 2 – Exploratory Data Analysis (EDA)
Uncovered outliers and missing values

Identified patterns in transaction amount, fraud, and product use

Summarized top insights in Jupyter Notebook

✅ Task 3 – Feature Engineering
Created:

Aggregate features (Total amount, Count, Avg)

Temporal features (Day, Hour, Month)

Encoded categorical variables

Normalized numerical data

Used xverse and woe for feature selection and scoring

✅ Task 4 – Proxy Variable Creation
RFM (Recency, Frequency, Monetary) engineered

Used KMeans clustering for customer segmentation

Defined is_high_risk = 1 for the lowest value cluster

✅ Task 5 – Model Training and Evaluation
Trained multiple models: Logistic Regression, Gradient Boosting

Evaluated using ROC-AUC, F1, Accuracy

Tracked experiments using MLflow

Registered the best model to MLflow Model Registry

Created automated unit tests with pytest

✅ Task 6 – Model Deployment & CI/CD
Deployed prediction API using FastAPI

Dockerized app with Dockerfile and docker-compose.yml

CI pipeline runs on every push:

flake8 for linting

pytest for test coverage

/predict endpoint accepts new customer data and returns:

Risk probability

Credit score (scaled)

Loan amount and duration recommendations

🧠 Learning Outcomes
Tools & Skills Used:
scikit-learn, mlflow, pytest, FastAPI, Docker

CI/CD with GitHub Actions

Model management with MLflow

Python best practices: pipelining, modular code, testing

Real-World Takeaways:
End-to-end ML systems must be automated, interpretable, and deployable

Proxy targets enable learning without perfect ground-truth

Regulatory alignment is as important as predictive power

🚀 How to Run the Project
🔧 Local API Setup
bash
Copy
Edit
git clone https://github.com/Yeabtssega/Credit-risk-model.git
cd Credit-risk-model
docker-compose up --build
🔍 Test API
Visit: http://127.0.0.1:8000/docs
Use Swagger UI to test the /predict endpoint.

📚 References
Statistical Models for Credit Scoring

Credit Scoring Guidelines - World Bank

Basel II Accord Summary

Weight of Evidence & IV Explanation

✅ Final Submission
🎯 GitHub Repo: github.com/Yeabtssega/Credit-risk-model
📦 Submission: July 1, 2025, 8PM UTC

