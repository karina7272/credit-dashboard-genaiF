
import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import base64

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("student_credit_data.csv")

# Save a copy of StudentID for later
student_ids = df['StudentID']

# Preprocessing (assume it's already done before model training)
X = df.drop(columns=['StudentID', 'CreditScore'])  # features
y = df['CreditScore']

# SHAP explainer
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# Streamlit app UI
st.title("🎓 GenAI Credit Scoring Dashboard")

# Filter Section
st.sidebar.header("🔍 SHAP Feature Filter")
all_features = list(X.columns)
selected_features = st.sidebar.multiselect("Select Features to Visualize:", all_features, default=all_features)

# SHAP Visualization Section
st.markdown("## 📊 SHAP Feature Impact Visualization")
with st.spinner("Generating SHAP chart..."):
    fig, ax = plt.subplots()
    if selected_features:
        shap.summary_plot(shap_values[:, selected_features], X[selected_features], show=False)
    else:
        shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

# Table preview
st.markdown("## 🧾 Student Data Preview")
st.dataframe(pd.concat([student_ids, X], axis=1).head(50))

# SHAP Interpretation
st.markdown("## 📘 SHAP Feature Impact Interpretation")
st.write("""
The SHAP summary plot offers a rich view into the decision logic of our fairness-aware academic credit scoring model. 
You can filter by features on the right sidebar. If no features are selected, all variables will be included.
""")
