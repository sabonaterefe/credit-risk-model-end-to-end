import streamlit as st
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from src.models.predict_model import predict_risk

# 📦 Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 🎯 Load model and pipeline
model = joblib.load("models/final_cv_model.pkl")
pipeline = joblib.load("models/fitted_pipeline.pkl")

# 📊 Load dataset to extract dropdown options
@st.cache_data
def load_dropdown_options():
    df = pd.read_csv("data/processed/cleaned_transactions.csv")
    return {
        "customers": sorted(df["CustomerId"].dropna().unique().tolist()),
        "channels": sorted(df["ChannelId"].dropna().unique().tolist()),
        "providers": sorted(df["ProviderId"].dropna().unique().tolist())
    }

options = load_dropdown_options()

# 🖼️ Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🎨 Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/credit-card.png", width=80)
    st.title("💼 Credit Risk App")
    st.markdown(
        """
        This app predicts whether a customer is **high risk** based on their transaction behavior.

        🔍 Powered by XGBoost  
        🧠 Trained on RFM features  
        🐍 Built with FastAPI + Streamlit  
        """
    )
    st.markdown("---")
    st.caption("Developed by Sabona • 2025")

# 🧾 Main UI
st.title("💳 Credit Risk Prediction")
st.markdown("Fill in the transaction details below to assess risk.")

prediction_result = None
input_df = None

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
        value="2018-11-15 02:18:49+00:00",
        placeholder="YYYY-MM-DD HH:MM:SS+00:00"
    )

    submitted = st.form_submit_button("🔮 Predict Risk")

if submitted:
    with st.spinner("🔍 Analyzing transaction and scoring risk..."):
        input_df = pd.DataFrame([{
            "Amount": amount,
            "Value": value,
            "ProductCategory": product_category,
            "ChannelId": channel_id,
            "ProviderId": provider_id,
            "CustomerId": customer_id,
            "TransactionStartTime": txn_time
        }])

        # Ensure numeric fields are valid
        input_df["Amount"] = pd.to_numeric(input_df["Amount"], errors="coerce")
        input_df["Value"] = pd.to_numeric(input_df["Value"], errors="coerce")

        if input_df[["Amount", "Value"]].isnull().any().any():
            st.error("❌ Invalid numeric input. Please check 'Amount' and 'Value'.")
        else:
            try:
                X = pipeline.transform(input_df)
                label, proba = predict_risk(model, X)
                prediction_result = (label[0], proba[0])
            except Exception as e:
                st.exception(f"Prediction failed: {e}")

# 🎯 Display results after form
if prediction_result:
    label, proba = prediction_result
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if label == 1:
        st.markdown(
            "<div style='background-color:#ffe6e6;padding:20px;border-radius:10px;'>"
            "<h3 style='color:#cc0000;'>🚨 High Risk Customer</h3>"
            f"<p style='font-size:18px;'>This customer has a <strong>{proba*100:.4f}%</strong> probability of being high risk.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#e6ffe6;padding:20px;border-radius:10px;'>"
            "<h3 style='color:#006600;'>✅ Low Risk Customer</h3>"
            f"<p style='font-size:18px;'>This customer has a <strong>{proba*100:.4f}%</strong> probability of being high risk.</p>"
            "</div>",
            unsafe_allow_html=True
        )

    st.metric("📈 Risk Probability", f"{proba:.6f}")

    with st.expander("🔍 View Input Details"):
        st.dataframe(input_df.astype(str).T.rename(columns={0: "Value"}))

    st.caption(f"🕒 Prediction generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("---")
    st.info("Was this prediction helpful?")
    col_yes, col_no = st.columns(2)
    with col_yes:
        st.button("👍 Yes")
    with col_no:
        st.button("👎 No")
