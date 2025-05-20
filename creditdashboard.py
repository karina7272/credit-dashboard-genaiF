
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import openai

# Secure API Key for GPT
openai.api_key = st.secrets["openai_api_key"]

# Page setup
st.set_page_config(page_title="GenAI Credit Scoring Dashboard", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1581091870622-1e7d88469d2d?fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 GenAI Academic Credit Scoring Dashboard")

uploaded_file = st.file_uploader("📁 Upload Your Student Credit CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Creditworthy' not in df.columns:
        df['Creditworthy'] = (
            (df['GPA'] > 2.5) &
            (df['RentPaidOnTime'] == 1) &
            (df['MissedPayments'] <= 1) &
            (df['CreditUtilization(%)'] < 60) &
            (df['FinancialLiteracyScore'] >= 70)
        ).astype(int)

    df_encoded = pd.get_dummies(df, columns=["Gender", "Race"], drop_first=True)
    base_features = [
        'Age', 'GPA', 'RentPaidOnTime', 'GigIncomeMonthly', 'CreditUtilization(%)',
        'MissedPayments', 'StudentLoans', 'PartTimeJob', 'FinancialLiteracyScore'
    ]
    encoded_features = [col for col in df_encoded.columns if col.startswith("Gender_") or col.startswith("Race_")]
    features = base_features + encoded_features

    scaler = StandardScaler()
    X = df_encoded[features]
    y = df_encoded['Creditworthy']
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    df['Prediction'] = model.predict(X_scaled)
    df['Confidence'] = (model.predict_proba(X_scaled)[:, 1] * 100).round(2)

    summaries = []
    hashes = []

    st.subheader("📜 GPT-Generated Credit Summaries")

    for i, row in df.iterrows():
        summary_input = f"""
        Student with GPA {row['GPA']}, credit utilization {row['CreditUtilization(%)']}%, 
        and financial literacy score {row['FinancialLiteracyScore']} is predicted to be 
        {'CREDITWORTHY' if row['Prediction'] == 1 else 'NOT CREDITWORTHY'} with confidence {row['Confidence']}%.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a credit risk analyst creating summaries."},
                    {"role": "user", "content": f"Write a professional credit summary:\n{summary_input}"}
                ]
            )
            gpt_summary = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            gpt_summary = f"{summary_input} [GPT unavailable: {str(e)}]"

        hash_string = f"{row['StudentID']}-{row['GPA']}-{row['CreditUtilization(%)']}-{row['FinancialLiteracyScore']}"
        hash_val = hashlib.sha256(hash_string.encode()).hexdigest()

        summaries.append(gpt_summary)
        hashes.append(hash_val)

    df['GPT_Summary'] = summaries
    df['Blockchain_Hash'] = hashes

    st.dataframe(df[['StudentID', 'Prediction', 'Confidence', 'GPT_Summary', 'Blockchain_Hash']])

    csv_export = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Results as CSV", data=csv_export, file_name="credit_scoring_results.csv", mime="text/csv")

    st.success("✅ Analysis complete! CSV export is ready.")

    # SHAP Explainability
    st.subheader("🔍 SHAP Feature Impact Visualization")
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X, feature_names=features, show=False)
    st.pyplot(fig)

    # Fairness-aware comparison
    st.subheader("⚖️ Fairness-Aware Model Report (No Race/Gender)")
    fair_features = [col for col in features if not ("Race_" in col or "Gender_" in col)]
    X_fair = df_encoded[fair_features]
    X_fair_scaled = scaler.fit_transform(X_fair)

    fair_model = LogisticRegression(max_iter=1000)
    fair_model.fit(X_fair_scaled, y)
    fair_preds = fair_model.predict(X_fair_scaled)

    report = classification_report(y, fair_preds, output_dict=False)
    st.text(report)

else:
    st.info("Please upload a CSV file to begin.")
