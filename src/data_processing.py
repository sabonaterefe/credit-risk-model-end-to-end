import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw/data.csv"
RAW_DEFINITIONS_PATH = "data/raw/Xente_Variable_Definitions.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_transactions.csv"
PROCESSED_DEFINITIONS_PATH = "data/processed/cleaned_variable_definitions.csv"

def load_data(data_path: str, definitions_path: str):
    print(f"📥 Loading transaction data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"📘 Loading variable definitions from {definitions_path}")
    definitions = pd.read_csv(definitions_path)

    # Fix encoding issues
    definitions['Column Name'] = definitions['Column Name'].astype(str).str.encode('utf-8', 'ignore').str.decode('utf-8')
    definitions['Definition'] = definitions['Definition'].astype(str).str.encode('utf-8', 'ignore').str.decode('utf-8')

    return df, definitions

def validate_schema(df: pd.DataFrame, definitions: pd.DataFrame):
    expected_columns = definitions['Column Name'].tolist()
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Missing columns in dataset: {missing_cols}")
    else:
        print("✅ All expected columns are present.")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Starting preprocessing...")

    df = df.drop_duplicates()
    print("✅ Dropped duplicate rows")

    # Convert TransactionStartTime to datetime
    df.loc[:, 'TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    print("🕒 Converted TransactionStartTime to datetime")

    # Extract datetime features safely and cast to integer
    df.loc[:, 'TransactionHour'] = df['TransactionStartTime'].apply(lambda x: x.hour if pd.notnull(x) else -1).astype(int)
    df.loc[:, 'TransactionDay'] = df['TransactionStartTime'].apply(lambda x: x.day if pd.notnull(x) else -1).astype(int)
    df.loc[:, 'TransactionWeekday'] = df['TransactionStartTime'].apply(lambda x: x.weekday() if pd.notnull(x) else -1).astype(int)
    print("📆 Extracted hour, day, and weekday from TransactionStartTime")

    # Fill missing categorical values
    categorical_cols = [
        'BatchId', 'CurrencyCode', 'CountryCode', 'ProviderId',
        'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna('Unknown')
            print(f"🔤 Filled missing values in {col} with 'Unknown'")

    # Fill missing numerical values and cap outliers
    numeric_cols = ['Amount', 'Value']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df.loc[:, col] = df[col].fillna(median_val)
            print(f"🔢 Filled missing values in {col} with median: {median_val}")

            # Cap outliers using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df.loc[:, col] = np.clip(df[col], lower, upper)
            print(f"📉 Capped outliers in {col} to [{lower:.2f}, {upper:.2f}]")

    # Fill missing FraudResult with mode
    if 'FraudResult' in df.columns and df['FraudResult'].isnull().sum() > 0:
        mode_val = df['FraudResult'].mode()[0]
        df.loc[:, 'FraudResult'] = df['FraudResult'].fillna(mode_val)
        print(f"🛡️ Filled missing FraudResult with mode: {mode_val}")

    print("✅ Preprocessing complete")
    return df

def save_outputs(df: pd.DataFrame, definitions: pd.DataFrame):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    definitions.to_csv(PROCESSED_DEFINITIONS_PATH, index=False)
    print(f"💾 Cleaned transactions saved to {PROCESSED_DATA_PATH}")
    print(f"📘 Cleaned variable definitions saved to {PROCESSED_DEFINITIONS_PATH}")

def main():
    df, definitions = load_data(RAW_DATA_PATH, RAW_DEFINITIONS_PATH)
    validate_schema(df, definitions)
    df_cleaned = preprocess_data(df)
    save_outputs(df_cleaned, definitions)

if __name__ == "__main__":
    main()
