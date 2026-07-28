import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="FraudGuard Africa",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        font-size: 1.3rem;
    }
    .result-fraud {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        font-size: 1.3rem;
    }
    .result-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load("models/fraudguard_xgboost_full.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model()

# Header
st.markdown('<p class="main-header">🛡️ FraudGuard Africa</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Mobile Money Fraud Detection System</p>', unsafe_allow_html=True)

# Input Section
st.subheader("Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    step = st.number_input("Time Step", min_value=1, value=150)
    transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
    amount = st.number_input("Amount (₦)", min_value=0.0, value=250000.0, step=5000.0)

with col2:
    nameOrig = st.text_input("Origin Account", "C123456789")
    oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=400000.0)
    newbalanceOrig = st.number_input("Origin New Balance", min_value=0.0, value=150000.0)

with col3:
    nameDest = st.text_input("Destination Account", "C987654321")
    oldbalanceDest = st.number_input("Destination Old Balance", min_value=0.0, value=15000.0)
    newbalanceDest = st.number_input("Destination New Balance", min_value=0.0, value=265000.0)

st.write("")
analyze_btn = st.button("Analyze Transaction", use_container_width=True)

if analyze_btn:
    data = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": nameOrig,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    df = pd.DataFrame([data])

    # Feature Engineering
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]
    df["amount_to_oldbalance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-8)
    df["amount_to_newbalance_ratio"] = df["amount"] / (df["newbalanceOrig"] + 1e-8)
    df["hour"] = df["step"] % 24
    df["is_night"] = df["hour"].isin([0,1,2,3,4,5,22,23]).astype(int)

    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    probability = float(model.predict_proba(df)[0][1])
    is_fraud = bool(model.predict(df)[0])

    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    st.markdown("---")
    st.subheader("Analysis Result")

    if risk_level == "High" or is_fraud:
        st.markdown('<div class="result-fraud">🚨 High Risk of Fraud Detected</div>', unsafe_allow_html=True)
    elif risk_level == "Medium":
        st.markdown('<div class="result-medium">⚠️ Medium Risk Transaction</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-safe">✅ Transaction Appears Safe</div>', unsafe_allow_html=True)

    st.write("")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud Probability", f"{probability*100:.3f}%")
    m2.metric("Risk Level", risk_level)
    m3.metric("Amount", f"₦{amount:,.0f}")
    m4.metric("Time", datetime.now().strftime("%H:%M:%S"))

    st.markdown("### Transaction Summary")
    st.markdown(f"""
    - **Type**: `{transaction_type}`
    - **Amount**: ₦{amount:,.2f}
    - **Origin Account**: {nameOrig}
    - **Origin Balance**: ₦{oldbalanceOrg:,.0f} → ₦{newbalanceOrig:,.0f}
    - **Destination Account**: {nameDest}
    - **Risk Level**: **{risk_level}**
    """)