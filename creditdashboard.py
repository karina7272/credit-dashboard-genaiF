import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import openai

openai.api_key = st.secrets["openai_api_key"]

st.set_page_config(page_title="GenAI Credit Scoring Dashboard", layout="wide", page_icon="📊")
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

    summaries, hashes = [], []

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

        hash_val = hashlib.sha256(f"{row['StudentID']}-{row['GPA']}-{row['CreditUtilization(%)']}-{row['FinancialLiteracyScore']}".encode()).hexdigest()
        summaries.append(gpt_summary)
        hashes.append(hash_val)

    df['GPT_Summary'] = summaries
    df['Blockchain_Hash'] = hashes

    st.subheader("📜 GPT-Generated Credit Summaries")
    st.dataframe(df[['StudentID', 'Prediction', 'Confidence', 'GPT_Summary', 'Blockchain_Hash']])
    csv_export = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv_export, file_name="credit_scoring_results.csv", mime="text/csv")

    # SHAP summary chart (original)
    st.subheader("🔍 SHAP Feature Impact Visualization")
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X, feature_names=features, show=False)
    st.pyplot(fig)

    # Interactive Plotly chart
    st.subheader("📊 Interactive SHAP Feature Importance")
    shap_df = pd.DataFrame(shap_values.values, columns=features)
    shap_mean = shap_df.abs().mean().sort_values(ascending=False)
    plot_df = pd.DataFrame({
        "Feature": shap_mean.index,
        "Mean |SHAP value|": shap_mean.values
    }).head(10)

    fig_plotly = px.bar(plot_df, x="Mean |SHAP value|", y="Feature", orientation="h")
    fig_plotly.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_plotly, use_container_width=True)
else:
    st.info("Please upload a CSV file to begin.")