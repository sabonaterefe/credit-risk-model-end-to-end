from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.features.transformers import AggregateCustomerFeatures, DateFeatureExtractor

def build_feature_pipeline():
    categorical_cols = ["ProductCategory", "ChannelId", "ProviderId"]
    numeric_cols = ["Amount", "Value", "TotalAmount", "AvgAmount", "TxnCount", "AmountStd"]

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    full_pipeline = Pipeline([
        ("date_features", DateFeatureExtractor()),
        ("aggregate_features", AggregateCustomerFeatures()),
        ("preprocessor", ColumnTransformer([
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numeric_cols)
        ]))
    ])

    return full_pipeline
