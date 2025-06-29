#This repository contains a full-stack implementation of a credit risk scoring system using alternative behavioral data from an eCommerce platform. The project is developed in collaboration with Bati Bank and Xente.

Project Objective:
To build, deploy, and automate a machine learning pipeline that predicts the creditworthiness of customers using behavioral transaction data. The system enables a Buy-Now-Pay-Later (BNPL) service by assigning risk probabilities, credit scores, and optimal loan terms to new applicants.

Business Context:
Bati Bank is partnering with a fast-growing eCommerce company to offer credit-based purchases. Traditional credit scoring methods are not feasible due to the lack of formal credit history. Instead, we leverage Recency, Frequency, and Monetary (RFM) patterns from transaction data to engineer a proxy for credit risk.

Features:
Proxy variable creation for credit default classification

Feature engineering from behavioral data

Risk probability prediction using ML models

Credit score generation from risk estimates

Loan amount and duration optimization

FastAPI-based scoring API

CI/CD with GitHub Actions

Model tracking with MLflow

Containerization with Docker

Credit Scoring Business Understanding
1. Basel II and Model Interpretability:
Basel II requires financial institutions to quantify and justify credit risk. This mandates interpretable, auditable models that regulators and stakeholders can trust.

2. Why Use a Proxy Variable?:
In the absence of a direct "default" label, we engineer a proxy using behavioral signals. This enables supervised learning but introduces risks if the proxy poorly reflects true creditworthiness.

3. Model Trade-offs:
Simple models (e.g., Logistic Regression with WoE) offer transparency but may lack predictive power. Complex models (e.g., Gradient Boosting) perform better but are harder to interpret. A hybrid approach balances compliance and performance.

Tech Stack:
Python 3.10+

Scikit-learn, XGBoost

FastAPI

MLflow

Docker & Docker Compose

GitHub Actions (CI/CD)

Pandas, NumPy, Matplotlib, Seaborn
