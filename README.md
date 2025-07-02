---
title: Credit Risk App
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: End-to-end ML credit risk prediction using behavioral data
license: MIT
---
=======
#This repository contains a full-stack implementation of a credit risk scoring system using alternative behavioral data from an eCommerce platform. The project is developed in collaboration with Bati Bank and Xente.
>>>>>>> 9c009cc (Update README.md)

# Credit Risk Model — End to End

This project delivers a full-stack credit scoring system using alternative behavioral data from an eCommerce platform. Built as part of the 10 Academy AI Mastery program in collaboration with Bati Bank and Xente, it enables Buy-Now-Pay-Later (BNPL) services by predicting customer creditworthiness.

---

## Objective

To build, deploy, and automate a machine learning pipeline that:
- Predicts credit risk using behavioral transaction data
- Generates credit scores and risk probabilities
- Optimizes loan terms for new applicants

---

## Business Context

Traditional credit scoring fails in low-data environments. We use Recency, Frequency, and Monetary (RFM) patterns to engineer a proxy for credit default, enabling supervised learning in the absence of formal credit history.

---

## Key Features

- Proxy variable creation from behavioral data
- Feature engineering and model training
- Risk probability and credit score generation
- FastAPI backend for real-time scoring
- Streamlit frontend for user interaction
- CI/CD with GitHub Actions
- MLflow for model tracking
- Dockerized for Hugging Face deployment

---

## Credit Scoring Principles

1. Basel II Compliance: Models must be interpretable, auditable, and regulator-friendly.
2. Proxy Variables: Behavioral proxies (RFM) substitute for missing default labels.
3. Model Trade-offs: Logistic Regression (WoE) for transparency; XGBoost for performance. A hybrid approach balances both.

---

##  Tech Stack

| Layer         | Tools Used                                      |
|---------------|--------------------------------------------------|
| Language      | Python 3.10+                                     |
| ML Libraries  | Scikit-learn, XGBoost                            |
| Backend       | FastAPI                                          |
| Frontend      | Streamlit                                        |
| MLOps         | MLflow, GitHub Actions                           |
| Deployment    | Docker, Docker Compose, Hugging Face Spaces      |
| Data Tools    | Pandas, NumPy, Matplotlib, Seaborn               |

---

## How It Works

1. User Input: Streamlit UI collects applicant data
2. API Call: FastAPI receives input and returns predictions
3. Model Inference: ML model estimates risk and credit score
4. Output: Streamlit displays results with loan recommendations

---

## Customize the App

- Modify `/src/streamlit_app.py` for UI
- Update `/src/api/main.py` for API logic
- Adjust `start.sh` and `Dockerfile` for deployment

---

## License

MIT

---

## Live Demo

👉 [Launch on Hugging Face](https://sabona333-credit-risk-app.hf.space)  
👉 [API Docs](https://sabona333-credit-risk-app.hf.space/api/docs)

---

