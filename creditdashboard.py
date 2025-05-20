
import streamlit as st
import pandas as pd
import plotly.express as px

# Existing parts of your app remain unchanged...

# New section: Interactive Feature Impact Visualization
st.markdown("## 🎯 Interactive Feature Impact Visualization")

# Load the CSV file
csv_file = "Simulated_Student_Credit_Data.csv"
try:
    df = pd.read_csv(csv_file)

    # List of filterable features
    filter_columns = [
        'StudentID', 'Gender', 'Race', 'Age', 'GPA', 'RentPaidOnTime',
        'GigIncomeMonthly', 'CreditUtilization(%)', 'MissedPayments',
        'StudentLoans', 'PartTimeJob', 'FinancialLiteracyScore'
    ]

    # Sidebar filters
    st.sidebar.header("🔎 Filter Data")
    selected_filters = {}
    for col in filter_columns:
        if df[col].dtype == 'object':
            selected = st.sidebar.multiselect(f"{col}:", options=df[col].unique())
            if selected:
                selected_filters[col] = selected
        else:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected = st.sidebar.slider(f"{col}:", min_val, max_val, (min_val, max_val))
            selected_filters[col] = selected

    # Apply filters
    filtered_df = df.copy()
    for col, selected in selected_filters.items():
        if isinstance(selected, list):
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]
        else:
            filtered_df = filtered_df[(filtered_df[col] >= selected[0]) & (filtered_df[col] <= selected[1])]

    # Plot
    st.subheader("📊 Feature Impact: GPA vs Credit Utilization (Filtered)")
    fig = px.scatter(
        filtered_df,
        x='GPA',
        y='CreditUtilization(%)',
        color='FinancialLiteracyScore',
        size='GigIncomeMonthly',
        hover_data=filter_columns
    )
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error(f"CSV file '{csv_file}' not found. Please upload the file to the directory.")
