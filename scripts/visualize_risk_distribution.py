import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    df = pd.read_csv("data/predictions/predictions.csv")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["risk_probability"], bins=30, kde=True, color="darkred")
    plt.title("Distribution of Risk Probabilities")
    plt.xlabel("Risk Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()

    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/risk_distribution.png")
    print("✅ Risk distribution plot saved to reports/figures/risk_distribution.png")

if __name__ == "__main__":
    main()
