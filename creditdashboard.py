import streamlit as st
import pandas as pd
import shap
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# Load the model and explainer
model = joblib.load("/mnt/data/credit_model.pkl")
explainer = joblib.load("/mnt/data/shap_explainer.pkl")

# Load data
df = pd.read_csv("/mnt/data/Simulated_Student_Credit_Data.csv")
X = df.drop(columns=["StudentID", "Prediction", "Confidence", "Blockchain_Hash", "GPT_Summary"])
y = df["Prediction"]

# Title
st.title("📊 GenAI Academic Credit Scoring Dashboard")

# Student Summary Table
st.subheader("📄 GPT-Generated Credit Summaries")
st.dataframe(df[["StudentID", "Prediction", "Confidence", "GPT_Summary", "Blockchain_Hash"]])

# SHAP Summary Plot (Static)
st.subheader("🔍 SHAP Feature Impact Visualization")
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="dot", show=False)
st.pyplot(plt.gcf())
plt.clf()

# Interactive SHAP Feature Importance (Newly Added)
st.markdown("### Interactive SHAP Feature Importance")
mean_shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_SHAP": abs(shap_values).mean(axis=0)
}).sort_values("Mean_SHAP", ascending=False)

selected_feature = st.selectbox("Select a SHAP Feature to Visualize", options=["All"] + list(mean_shap_df["Feature"]))

if selected_feature != "All":
    selected_data = df.copy()
    fig = px.bar(mean_shap_df[mean_shap_df["Feature"] == selected_feature],
                 x="Mean_SHAP", y="Feature", orientation="h",
                 title=f"Impact of Feature: {selected_feature}", height=300)
else:
    fig = px.bar(mean_shap_df.head(15),
                 x="Mean_SHAP", y="Feature", orientation="h",
                 title="Top 15 Features by SHAP Impact", height=700)

st.plotly_chart(fig, use_container_width=True)