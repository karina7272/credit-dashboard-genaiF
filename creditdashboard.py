
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Sample data simulation (replace with your own DataFrame)
@st.cache_data
def load_data():
    df = pd.DataFrame({
        "StudentID": [f"SID{i:05d}" for i in range(100)],
        "Gender": pd.Categorical(["Male", "Female", "Non-binary"] * 34 + ["Female"]),
        "Race": pd.Categorical(["White", "Black", "Hispanic", "Asian", "Other"] * 20),
        "Age": pd.Series(range(18, 118)),
        "GPA": pd.Series([round(2.0 + i * 0.02, 2) for i in range(100)]),
        "RentPaidOnTime": pd.Series([i % 2 for i in range(100)]),
        "GigIncomeMonthly": pd.Series([i * 10 for i in range(100)]),
        "CreditUtilization(%)": pd.Series([round(20 + i * 0.5, 2) for i in range(100)]),
        "MissedPayments": pd.Series([i % 4 for i in range(100)]),
        "StudentLoans": pd.Series([i % 2 for i in range(100)]),
        "PartTimeJob": pd.Series([i % 2 for i in range(100)]),
        "FinancialLiteracyScore": pd.Series([i % 10 for i in range(100)]),
        "Default": pd.Series([i % 2 for i in range(100)])  # target
    })
    return df

df = load_data()

# Encode categorical variables for modeling
X = pd.get_dummies(df.drop(columns=["StudentID", "Default"]), drop_first=True)
y = df["Default"]

# Train a basic model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)[1]

# Sidebar filter
default_features = [
    "StudentID", "Gender", "Race", "Age", "GPA", "RentPaidOnTime",
    "GigIncomeMonthly", "CreditUtilization(%)", "MissedPayments",
    "StudentLoans", "PartTimeJob", "FinancialLiteracyScore"
]

selected_features = st.multiselect(
    "Select features to display in SHAP summary",
    options=list(X.columns),
    default=[col for col in X.columns if any(df_name in col for df_name in default_features)]
)

# Filter data and SHAP values for visualization
selected_indices = [X.columns.get_loc(c) for c in selected_features if c in X.columns]
shap_values_filtered = shap_values[:, selected_indices]
X_filtered = X[selected_features]

# SHAP Summary Plot
st.subheader("🔍 SHAP Feature Impact Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values_filtered, X_filtered, show=False)
st.pyplot(fig)

# Optional: Show filtered data
with st.expander("Preview Feature Data"):
    st.dataframe(X_filtered)
