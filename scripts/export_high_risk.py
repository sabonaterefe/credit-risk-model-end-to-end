import pandas as pd
import os

def main(threshold=0.5):
    df = pd.read_csv("data/predictions/predictions.csv")
    high_risk = df[df["risk_probability"] > threshold]

    os.makedirs("data/predictions", exist_ok=True)
    high_risk.to_csv("data/predictions/high_risk_customers.csv", index=False)
    print(f"✅ Exported {len(high_risk)} high-risk customers to data/predictions/high_risk_customers.csv")

if __name__ == "__main__":
    main()
