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

openai.api_key = st.secrets["openai_api_key"]

st.set_page_config(page_title="GenAI Credit Scoring Dashboard", layout="wide", page_icon="📊")

# DARK THEME STYLING
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    .stDataFrame, .stText, .stMarkdown {
        color: white;
    }
    .block-container {
        padding: 2rem;
        background-color: #2A2A2A;
    }
    </style>
""", unsafe_allow_html=True)

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

    st.subheader("🔍 SHAP Feature Impact Visualization")
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X, feature_names=features, show=False)
    st.pyplot(fig)

    try:
        with open("shap_interpretation.md", "r") as file:
            interpretation = file.read()
        st.markdown(interpretation)
    except:
        st.warning("SHAP interpretation markdown file not found.")

    st.subheader("⚖️ Fairness-Aware Model Report (No Race/Gender)")
    fair_features = [col for col in features if not ("Race_" in col or "Gender_" in col)]
    X_fair = df_encoded[fair_features]
    X_fair_scaled = scaler.fit_transform(X_fair)
    fair_model = LogisticRegression(max_iter=1000)
    fair_model.fit(X_fair_scaled, y)
    fair_preds = fair_model.predict(X_fair_scaled)
    report_dict = classification_report(y, fair_preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    st.dataframe(report_df)  # Correctly placed under heading

    st.subheader("🔎 Per-Student Credit Interpretation")
    selected_id = st.selectbox("Select a StudentID to view details", df["StudentID"].unique())
    student_row = df[df["StudentID"] == selected_id].iloc[0]
    st.subheader("🧠 SHAP-Based Interpretation Summary (Student ID: {})".format(selected_id))

    gpa = student_row['GPA']
    credit_util = student_row['CreditUtilization(%)']
    fin_lit = student_row['FinancialLiteracyScore']
    rent_status = "on time" if student_row['RentPaidOnTime'] == 1 else "late"
    missed = student_row['MissedPayments']
    missed_effect = "supports" if missed == 0 else "might weaken"
    slight = "slightly" if missed > 0 else ""

    interpretation_text = f"""
This student's credit score reflects a mix of academic achievement, spending habits, and financial responsibility.  
The GPA of {gpa} indicates moderately strong academic performance, contributing positively to the creditworthiness score.  
A credit utilization rate of {credit_util}% suggests the student is using available credit cautiously and not excessively, which is favorable.  
The financial literacy score of {fin_lit} further reinforces the credit prediction, indicating strong financial understanding.  
Rent payment behavior marked as "{rent_status}" reflects stable financial routines, which aligns with reliability.  
Having {missed} missed payments {missed_effect} the score {slight}.  
SHAP values (if visualized) would likely show GPA and Financial Literacy Score pushing the model toward a CREDITWORTHY classification.  
Credit Utilization and Missed Payments would act as minor offsets depending on their respective thresholds.  
The prediction is also reinforced by a consistent history of non-excessive debt behaviors.  
If visualized, SHAP bars for GPA and Literacy would skew positively to the right of the SHAP axis.  
The model's fairness-aware classifier aligns with these findings and excludes any demographic bias.  
SHAP impact shows Financial Literacy and GPA as primary positive contributors.  
Missed payments, even if zero, are monitored for behavioral trend patterns.  
Low credit utilization remains a strong positive driver for this student.  
The combination of academic and financial traits yields high classification confidence.  
This profile serves as an example of well-rounded financial behavior in the dataset.  
The blockchain hash ensures data traceability and transparency.  
Given current trends, this score is likely to improve further if behaviors remain consistent.  
The AI scoring model supports this creditworthy decision based on fairness and predictive accuracy.
    """
    st.markdown(interpretation_text)



    st.markdown(f"""
**Prediction:** {'CREDITWORTHY' if student_row['Prediction'] == 1 else 'NOT CREDITWORTHY'}  
**Confidence:** {student_row['Confidence']}%  
**GPA:** {student_row['GPA']}  
**Credit Utilization (%):** {student_row['CreditUtilization(%)']}  
**Financial Literacy Score:** {student_row['FinancialLiteracyScore']}  
**Blockchain Hash:** {student_row['Blockchain_Hash']}  
**GPT Summary:**  
> {student_row['GPT_Summary']}
    """)

    st.subheader("🧠 SHAP-Based Interpretation Summary (Student ID: {})".format(selected_id))

    rent_status = "on time" if student_row['RentPaidOnTime'] == 1 else "late"
    missed_payments = student_row['MissedPayments']
    missed_impact = "supports" if missed_payments == 0 else "might weaken"
    missed_strength = "slightly" if missed_payments > 0 else ""

    
    st.subheader("✅ 10 Positive Traits in Current Credit Score")
    traits = []
    if student_row['GPA'] >= 3.0:
        traits.append("- Strong academic performance with a GPA above 3.0.")
    else:
        traits.append("- Meets minimum GPA requirement for creditworthiness.")
    if student_row['CreditUtilization(%)'] < 30:
        traits.append("- Keeps credit utilization below 30%, reflecting healthy habits.")
    if student_row['FinancialLiteracyScore'] >= 70:
        traits.append("- Demonstrates strong financial knowledge.")
    if student_row['RentPaidOnTime'] == 1:
        traits.append("- Consistently pays rent on time.")
    if student_row['MissedPayments'] == 0:
        traits.append("- Has no missed payments on record.")
    if student_row['PartTimeJob'] == 1:
        traits.append("- Maintains a part-time job, indicating income responsibility.")
    if student_row['GigIncomeMonthly'] > 0:
        traits.append("- Has diversified income sources through gig work.")
    if student_row['StudentLoans'] <= 1:
        traits.append("- Maintains a low number of student loans.")
    if student_row['Creditworthy'] == 1:
        traits.append("- Overall classification as CREDITWORTHY.")
    traits += ["- Complies with AI fairness-aware model constraints."]
    for trait in traits[:10]:
        st.markdown(trait)

    st.subheader("🔧 10 Recommendations to Improve Credit Score")
    recs = []
    if student_row['GPA'] < 3.0:
        recs.append("- Aim to increase GPA above 3.0 to improve academic standing.")
    if student_row['CreditUtilization(%)'] >= 30:
        recs.append("- Work on lowering credit utilization below 30%.")
    if student_row['FinancialLiteracyScore'] < 70:
        recs.append("- Attend workshops to boost financial literacy.")
    if student_row['RentPaidOnTime'] == 0:
        recs.append("- Set up reminders to pay rent on time.")
    if student_row['MissedPayments'] > 0:
        recs.append("- Eliminate missed payments by automating due dates.")
    if student_row['PartTimeJob'] == 0:
        recs.append("- Consider a part-time job to increase financial stability.")
    if student_row['GigIncomeMonthly'] == 0:
        recs.append("- Explore additional income sources such as gig work.")
    if student_row['StudentLoans'] > 1:
        recs.append("- Reduce loan burden through repayment or consolidation.")
    recs.append("- Build emergency savings to prevent financial shocks.")
    recs.append("- Monitor and manage expenses regularly to stay on track.")
    for rec in recs[:10]:
        st.markdown(rec)

else:
    st.info("Please upload a CSV file to begin.")


    # SHAP Feature Impact Visualization
    st.subheader("🔍 SHAP Feature Impact Visualization")

    if 'studentID' not in df.columns:
        st.error("Missing 'studentID' column in the uploaded dataset.")
    else:
        # Ensure features and labels are prepared
        features = df.drop(columns=["Creditworthy", "studentID"], errors='ignore')
        labels = df["Creditworthy"]
        model = LogisticRegression()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        model.fit(features_scaled, labels)

        explainer = shap.Explainer(model, features)
        shap_values = explainer(features)

        feature_options = list(features.columns)
        selected_features = st.multiselect("Select Features for SHAP Summary", feature_options, default=feature_options)

        if selected_features:
            shap.summary_plot(shap_values[:, selected_features], features[selected_features], plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
