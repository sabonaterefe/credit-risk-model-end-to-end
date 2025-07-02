from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TransactionStartTime" in X.columns:
            X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"], errors="coerce")
            X["TransactionHour"] = X["TransactionStartTime"].dt.hour
            X["TransactionDay"] = X["TransactionStartTime"].dt.day
            X["TransactionMonth"] = X["TransactionStartTime"].dt.month
            X["TransactionYear"] = X["TransactionStartTime"].dt.year
            X = X.drop(columns=["TransactionStartTime"])
        return X

class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "CustomerId" not in X.columns or "Amount" not in X.columns:
            raise ValueError("Required columns 'CustomerId' and 'Amount' not found in input.")

        agg = X.groupby("CustomerId").agg({
            "Amount": ["sum", "mean", "count", "std"]
        }).reset_index()
        agg.columns = ["CustomerId", "TotalAmount", "AvgAmount", "TxnCount", "AmountStd"]

        return X.merge(agg, on="CustomerId", how="left")
