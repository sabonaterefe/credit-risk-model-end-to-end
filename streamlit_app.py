import os
import sys
import pandas as pd
import streamlit as st
from datetime import datetime

# ✅ Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# ✅ Import prediction functions
from src.models.predict_model import load_pipeline, predict_risk

# ✅ Load trained pipeline
pipeline = load_pipeline("models/fitted_pipeline.pkl")

# ✅ Load dropdown options from dataset
@st.cache_data
def load_dropdown_options():
    df = pd.read_csv("data/processed/cleaned_transactions.csv")
    return {
        "customers": sorted(df["CustomerId"].dropna().unique().tolist()),
        "channels": sorted(df["ChannelId"].dropna().unique().tolist()),
        "providers": sorted(df["ProviderId"].dropna().unique().tolist())
    }

options = load_dropdown_options()

# ✅ Page configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# ✅ Title and instructions
st.title("🔍 Credit Risk Prediction")
st.markdown("Enter transaction details below to evaluate the customer's credit risk.")

# ✅ Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
        product_category = st.selectbox("🛍️ Product Category", ["airtime", "loan", "data", "utility"])
        provider_id = st.selectbox("🏢 Provider ID", options["providers"])
    with col2:
        value = st.number_input("📦 Transaction Value", min_value=0.0, value=1000.0, step=100.0)
        channel_id = st.selectbox("📡 Channel ID", options["channels"])
        customer_id = st.selectbox("🧑 Customer ID", options["customers"])

    txn_time = st.text_input(
        "🕒 Transaction Start Time",
        value="2018-11-15 03:12:00+00:00",
        placeholder="YYYY-MM-DD HH:MM:SS+00:00"
    )

    submitted = st.form_submit_button("🔮 Predict Risk")

# ✅ Run prediction
if submitted:
    with st.spinner("Scoring risk..."):
        input_df = pd.DataFrame([{
            "Amount": amount,
            "Value": value,
            "ProductCategory": product_category,
            "ChannelId": channel_id,
            "ProviderId": provider_id,
            "CustomerId": customer_id,
            "TransactionStartTime": txn_time
        }])

        input_df["Amount"] = pd.to_numeric(input_df["Amount"], errors="coerce")
        input_df["Value"] = pd.to_numeric(input_df["Value"], errors="coerce")

        if input_df[["Amount", "Value"]].isnull().any().any():
            st.error("❌ Invalid numeric input. Please check 'Amount' and 'Value'.")
        else:
            try:
                label, proba, risk_band, top_features = predict_risk(pipeline, input_df)

                # ✅ Display results
                st.markdown("---")
                st.subheader("📊 Prediction Summary")

                risk_color = {
                    "Low": "#e6ffe6",
                    "Medium": "#fff8e6",
                    "High": "#ffe6e6"
                }
                text_color = {
                    "Low": "#006600",
                    "Medium": "#cc9900",
                    "High": "#cc0000"
                }

                st.markdown(
                    f"""
                    <div style='background-color:{risk_color[risk_band]};padding:20px;border-radius:10px;'>
                        <h3 style='color:{text_color[risk_band]};'>🧠 {risk_band} Risk Customer</h3>
                        <p style='font-size:18px;'>This customer has a <strong>{proba * 100:.2f}%</strong> probability of defaulting on credit.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric("📈 Risk Probability", f"{proba:.6f}")

                with st.expander("🔍 Top Contributing Features"):
                    for feat in top_features:
                        st.markdown(f"- **{feat['feature']}**: SHAP = `{feat['shap_value']:.4f}`")

                with st.expander("📋 Input Details"):
                    st.dataframe(input_df.astype(str).T.rename(columns={0: "Value"}))

                st.caption(f"🕒 Prediction generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                st.exception(f"Prediction failed: {e}")
