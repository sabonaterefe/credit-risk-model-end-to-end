import pandas as pd
import numpy as np
from src.models.predict_model import load_pipeline, predict_risk

def test_predict_risk_output_shape():
    pipeline = load_pipeline("models/fitted_pipeline.pkl")
    sample_input = pd.DataFrame([{
        "Amount": 1000.0,
        "Value": 100.0,
        "ProductCategory": "loan",
        "ChannelId": "ChannelId_1",
        "ProviderId": "ProviderId_2",
        "CustomerId": "CustomerId_123",
        "TransactionStartTime": "2018-11-15 03:12:00+00:00"
    }])
    label, proba, risk_band, top_features = predict_risk(pipeline, sample_input)

    assert isinstance(label, int)
    assert isinstance(proba, (float, np.floating))
    assert risk_band in ["Low", "Medium", "High"]
    assert isinstance(top_features, list)
    assert len(top_features) > 0
