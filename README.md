Credit Scoring Business Understanding

# Task 1 - Understanding Credit Risk

This section summarizes the business and regulatory context for building a credit scoring model using alternative data, based on the following references:

- https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf  
- https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf  
- https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf  
- https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03  
- https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/  
- https://www.risk-officer.com/Credit_Risk.htm  

 1. Basel II and Model Interpretability  
The Basel II Accord emphasizes the importance of quantifying and managing credit risk through internal models. It requires financial institutions to use models that are transparent, interpretable, and auditable. This means our credit scoring model must not only be accurate but also explainable to regulators and stakeholders. Interpretability ensures trust, supports compliance, and enables responsible lending decisions.

 2. Why Use a Proxy Variable?  
Since we lack a direct label for loan default, we must engineer a proxy variable—such as one based on behavioral patterns like Recency, Frequency, and Monetary (RFM) activity. This allows us to train a supervised model. However, using a proxy introduces risk: if the proxy poorly reflects true creditworthiness, the model may misclassify customers, leading to financial losses or unfair credit decisions. Careful validation and alignment with business goals are essential.

 3. Model Trade-offs in a Regulated Context  
Simple models like Logistic Regression with Weight of Evidence (WoE) are highly interpretable and align well with regulatory expectations. However, they may underperform on complex, nonlinear data. Advanced models like Gradient Boosting Machines (GBMs) offer better predictive power but are harder to explain. In regulated environments, the trade-off is between transparency and performance. A hybrid approach—using complex models internally and interpretable models for decision justification—can offer the best of both worlds.


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


#New commit in readme 




