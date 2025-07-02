from pydantic import BaseModel
import pandas as pd

class CustomerInput(BaseModel):
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    CustomerId: str
    TransactionStartTime: str

    def to_df(self):
        return pd.DataFrame([self.dict()])

class RiskPrediction(BaseModel):
    predicted_label: int
    risk_probability: float
