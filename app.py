
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# ---------------------------------
# 1. PAGE SETTINGS
# ---------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ---------------------------------
# 2. CUSTOM DESIGN
# ---------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}
[data-testid="stMetric"] {
    background: #1b2638;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #34445c;
}
h1, h2, h3 {
    color: #65b9ff;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection")
st.caption("Machine Learning | Isolation Forest | Transaction Monitoring")

# ---------------------------------
# 3. LOAD TRAINED MODEL FILES
# ---------------------------------
@st.cache_resource
def load_model_files():
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    model = joblib.load(BASE_DIR / "isolation_forest_model.pkl")

    feature_file = BASE_DIR / "feature_names.pkl"

    if feature_file.exists():
        features = joblib.load(feature_file)
    elif hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise FileNotFoundError(
            "feature_names.pkl is missing. "
            "Please put it in the project folder."
        )

    return scaler, model, list(features)


try:
    scaler, model, feature_names = load_model_files()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ---------------------------------
# 4. LOAD FRAUD DETECTION REPORT
# ---------------------------------
report_path = BASE_DIR / "fraud_detection_report.csv"

if report_path.exists():
    df = pd.read_csv(report_path)
else:
    st.warning("The fraud detection CSV report was not found.")
    uploaded_report = st.file_uploader(
        "Upload fraud_detection_report.csv",
        type=["csv"]
    )

    if uploaded_report is None:
        st.stop()

    df = pd.read_csv(uploaded_report)

df = df.reset_index(drop=True)

if df.empty:
    st.error("Your CSV report is empty.")
    st.stop()

# ---------------------------------
# 5. IDENTIFY FRAUD PREDICTIONS
# ---------------------------------
def get_fraud_mask(data):
    # Prefer an existing status column
    if "status" in data.columns:
        status = data["status"].astype(str).str.lower()
        return status.str.contains(
            "fraud|anomal|suspicious|critical",
            regex=True
        )

    # Otherwise use the prediction column
    prediction_col = next(
        (c for c in ["prediction", "predicted", "pred"]
         if c in data.columns),
        None
    )

    if prediction_col is not None:
        values = pd.to_numeric(
            data[prediction_col], errors="coerce"
        )

        # Isolation Forest: -1 means anomaly
        if set(values.dropna().unique()).issubset({-1, 1}):
            return values == -1

        # Converted predictions: 1 means potential fraud
        return values == 1

    return pd.Series(False, index=data.index)


fraud_mask = get_fraud_mask(df)
normal_count = int((~fraud_mask).sum())
fraud_count = int(fraud_mask.sum())
total_count = len(df)
fraud_rate = (fraud_count / total_count * 100) if total_count else 0

# ---------------------------------
# 6. SIDEBAR
# ---------------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Transaction Analytics",
        "Fraud Detection",
        "Fraud Alerts"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Model: Isolation Forest\n\n"
    "Prediction -1: Potential anomaly\n\n"
    "Prediction 1: Normal"
)

# ---------------------------------
# 7. OVERVIEW
# ---------------------------------
if page == "Overview":

    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_count:,}")
    col2.metric("Normal Transactions", f"{normal_count:,}")
    col3.metric("Potential Fraud", f"{fraud_count:,}")
    col4.metric("Anomaly Rate", f"{fraud_rate:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Distribution")

        pie_data = pd.DataFrame({
            "Category": ["Normal", "Potential Fraud"],
            "Count": [normal_count, fraud_count]
        })

        fig = px.pie(
            pie_data,
            names="Category",
            values="Count",
            hole=0.55,
            color="Category",
            color_discrete_map={
                "Normal": "#20c997",
                "Potential Fraud": "#ff4b4b"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Transaction Amount Distribution")

        if "Amount" in df.columns:
            amount_data = df.copy()
            amount_data["Detection Status"] = np.where(
                fraud_mask,
                "Potential Fraud",
                "Normal"
            )

            fig = px.histogram(
                amount_data,
                x="Amount",
                color="Detection Status",
                nbins=40,
                barmode="overlay",
                color_discrete_map={
                    "Normal": "#20c997",
                    "Potential Fraud": "#ff4b4b"
                }
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Amount column is not available in this report.")

    st.markdown("---")
    st.subheader("Recent Potential Fraud Transactions")

    alerts = df.loc[fraud_mask].copy()

    if not alerts.empty:
        st.dataframe(alerts.head(15), use_container_width=True)
    else:
        st.success("No potential fraud transactions in this report.")

# ---------------------------------
# 8. TRANSACTION ANALYTICS
# ---------------------------------
elif page == "Transaction Analytics":

    st.header("📊 Transaction Analytics")

    if "Amount" in df.columns:
        st.subheader("Transaction Amounts")

        fig = px.box(
            df.assign(
                Detection=np.where(
                    fraud_mask, "Potential Fraud", "Normal"
                )
            ),
            x="Detection",
            y="Amount",
            color="Detection",
            color_discrete_map={
                "Normal": "#20c997",
                "Potential Fraud": "#ff4b4b"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    if "Time" in df.columns:
        st.subheader("Transaction Activity Over Time")

        time_data = df.copy()
        time_data["Detection"] = np.where(
            fraud_mask, "Potential Fraud", "Normal"
        )

        fig = px.histogram(
            time_data,
            x="Time",
            color="Detection",
            nbins=40,
            color_discrete_map={
                "Normal": "#20c997",
                "Potential Fraud": "#ff4b4b"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Transaction Report")
    st.dataframe(df, use_container_width=True)

# ---------------------------------
# 9. INTERACTIVE FRAUD DETECTION
# ---------------------------------
elif page == "Fraud Detection":

    st.header("🔍 Test a Transaction")

    st.write(
        "Select a transaction from the report and let the trained "
        "Isolation Forest model examine it."
    )

    # Check whether report contains all model input features
    missing_features = [
        col for col in feature_names if col not in df.columns
    ]

    prediction_data = df

    if missing_features:
        st.warning(
            "Your report does not contain all the original model "
            "features. Upload a CSV containing the original transaction "
            "features to use interactive prediction."
        )

        uploaded_data = st.file_uploader(
            "Upload transaction data CSV",
            type=["csv"],
            key="prediction_csv"
        )

        if uploaded_data is None:
            st.info(
                "The report and analytics pages still work. "
                "For prediction, upload a CSV with all model features."
            )
            st.stop()

        prediction_data = pd.read_csv(uploaded_data).reset_index(drop=True)

        missing_features = [
            col for col in feature_names
            if col not in prediction_data.columns
        ]

        if missing_features:
            st.error(
                "Missing model features: " + ", ".join(missing_features)
            )
            st.stop()

    if prediction_data.empty:
        st.error("No transactions are available for prediction.")
        st.stop()

    row_number = st.number_input(
        "Enter transaction row number",
        min_value=0,
        max_value=len(prediction_data) - 1,
        value=0,
        step=1
    )

    selected = prediction_data.iloc[[int(row_number)]][feature_names]

    with st.expander("View selected transaction details"):
        st.dataframe(selected, use_container_width=True)

    if st.button("🚨 Detect Fraud", type="primary"):

        try:
            # Keep exactly the same feature order used in training
            selected_scaled = scaler.transform(selected)

            prediction = model.predict(selected_scaled)[0]

            st.markdown("---")

            if prediction == -1:
                st.error(
                    "🚨 POTENTIAL FRAUD / ANOMALY DETECTED!"
                )
                st.write(
                    "The model has identified unusual transaction "
                    "patterns. Further investigation is recommended."
                )
            else:
                st.success(
                    "✅ TRANSACTION APPEARS NORMAL"
                )
                st.write(
                    "The model did not identify this transaction "
                    "as an anomaly."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------
# 10. FRAUD ALERTS
# ---------------------------------
elif page == "Fraud Alerts":

    st.header("🚨 Potential Fraud Alerts")

    alerts = df.loc[fraud_mask].copy()

    st.metric("Potential Fraud Transactions", len(alerts))

    if not alerts.empty:
        st.dataframe(alerts, use_container_width=True)

        csv_data = alerts.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Fraud Alerts",
            data=csv_data,
            file_name="potential_fraud_alerts.csv",
            mime="text/csv"
        )
    else:
        st.success("No potential fraud alerts found.")

# ---------------------------------
# 11. DOWNLOAD FULL REPORT
# ---------------------------------
st.sidebar.markdown("---")

csv_data = df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    "⬇️ Download Full Report",
    data=csv_data,
    file_name="fraud_detection_report.csv",
    mime="text/csv"
)

st.sidebar.caption("Credit Card Fraud Detection Project")
