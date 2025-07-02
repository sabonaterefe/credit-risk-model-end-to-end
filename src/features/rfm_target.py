import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_rfm_features(df, snapshot_date):
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"]).dt.tz_localize(None)

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    return rfm

def assign_risk_label(rfm_df, n_clusters=3):
    rfm_df = rfm_df.copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_stats = rfm_df.groupby("Cluster")[["Frequency", "Monetary"]].mean()
    risk_cluster = cluster_stats.mean(axis=1).idxmin()

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == risk_cluster).astype(int)
    return rfm_df[["CustomerId", "is_high_risk"]]
