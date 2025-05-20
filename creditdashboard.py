import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Title
st.title("GenAI Academic Credit Scoring Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload Your Student Credit CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📜 GPT-Generated Credit Summaries")
    st.dataframe(df)

    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="credit_summary.csv")

    st.subheader("🔍 SHAP Feature Impact Visualization")

    column_to_plot = st.selectbox("Select a SHAP Feature to Visualize", df.columns[1:], index=1)

    if column_to_plot:
        shap_values = np.abs(df[[column_to_plot]]).mean().reset_index()
        shap_values.columns = ['Feature', 'Mean SHAP value']
        fig = px.bar(shap_values, x="Mean SHAP value", y="Feature", orientation='h',
                     title="Interactive SHAP Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Student Credit Interpretation")
    selected_id = st.selectbox("Select a Student ID", df["StudentID"].unique())
    selected_student = df[df["StudentID"] == selected_id].iloc[0]

    st.write(f"**Prediction:** {'CREDITWORTHY' if selected_student['Prediction'] == 1 else 'NOT CREDITWORTHY'}")
    st.write(f"**Confidence:** {selected_student['Confidence']}%")
    st.write(f"**GPA:** {selected_student['GPA']}")
    st.write(f"**Credit Utilization (%):** {selected_student['CreditUtilization(%)']}")
    st.write(f"**Financial Literacy Score:** {selected_student['FinancialLiteracyScore']}")
    st.write(f"**Blockchain Hash:** {selected_student['Blockchain_Hash']}")

    st.text_area("GPT Summary:", value=f"""Student with GPA {selected_student['GPA']}, credit utilization {selected_student['CreditUtilization(%)']}%,
and financial literacy score {selected_student['FinancialLiteracyScore']} is predicted to be
{'CREDITWORTHY' if selected_student['Prediction'] == 1 else 'NOT CREDITWORTHY'} with confidence {selected_student['Confidence']}%.
[GPT unavailable: demo mode]""", height=160)

    st.subheader(f"🧠 SHAP-Based Interpretation Summary (Student ID: {selected_id})")
    gpa = selected_student['GPA']
    util = selected_student['CreditUtilization(%)']
    lit = selected_student['FinancialLiteracyScore']
    rent = "on time" if selected_student['RentPaidOnTime'] == 1 else "late"
    missed = selected_student['MissedPayments']
    effect = "supports" if missed == 0 else "might weaken"
    slight = "" if missed == 0 else "slightly"

    st.markdown(f"""This student's credit score reflects a mix of academic achievement, spending habits, and financial responsibility.  
The GPA of {gpa} indicates moderately strong academic performance, contributing positively to the creditworthiness score.  
A credit utilization rate of {util}% suggests the student is using available credit cautiously and not excessively, which is favorable.  
The financial literacy score of {lit} further reinforces the credit prediction, indicating strong financial understanding.  
Rent payment behavior marked as '{rent}' reflects stable financial routines, which aligns with reliability.  
Having {missed} missed payments {effect} the score {slight}.  
SHAP values (if visualized) would likely show GPA and Financial Literacy Score pushing the model toward a CREDITWORTHY classification.  
Credit Utilization and Missed Payments would act as minor offsets depending on their respective thresholds.  
The prediction is also reinforced by a consistent history of non-excessive debt behaviors.  
This profile serves as an example of well-rounded financial behavior in the dataset.""") 

    st.subheader("✅ 10 Positive Traits in Current Credit Score")
    st.markdown("""- Maintains a reasonable GPA that meets minimum academic performance.  
- Keeps credit utilization under 30%, reflecting healthy credit habits.  
- Financial literacy score is above average, demonstrating knowledge.  
- Shows consistent rent payment history.  
- Limited or no frequent missed payments.  
- Possesses diversified financial traits beneficial for scoring.  
- Conforms to standard risk thresholds on major indicators.  
- Demonstrates discipline in credit management.  
- Aligns well with AI model’s fairness constraints.  
- Has an overall strong blockchain-verified credit profile.""") 

    st.subheader("🔧 10 Recommendations to Improve Credit Score")
    st.markdown("""- Improve GPA further to enhance long-term financial perceptions.  
- Continue lowering credit utilization toward 10–15%.  
- Attend more financial literacy workshops to raise scores.  
- Explore part-time job options for additional income sources.  
- Set automatic payment reminders to avoid any future lapses.  
- Build savings history to supplement credit metrics.  
- Maintain zero missed payments consistently.  
- Reduce total loan balances if applicable.  
- Engage in responsible credit-building practices monthly.  
- Diversify financial responsibilities gradually and smartly.""")
else:
    st.warning("Please upload a CSV file to begin.")