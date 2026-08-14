import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---- Page setup ----
st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Dropout Risk Predictor")
st.write(
    "Enter a student's details below to estimate their risk of dropping out, "
    "staying enrolled, or graduating — and see which factors influenced the prediction."
)

# ---- Load the trained model (this file must sit in the same folder as app.py) ----
model = joblib.load("dropout_model.pkl")

# The exact 12 features the model was trained on, in this exact order
FEATURE_ORDER = [
    "Curricular units 2nd sem (approved)",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (enrolled)",
    "Course",
    "Age at enrollment",
    "Gender",
    "Unemployment rate",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (grade)",
    "Debtor",
    "Scholarship holder",
]

TARGET_LABELS = ["Dropout", "Enrolled", "Graduate"]

# ---- Input form ----
st.subheader("Student Details")

col1, col2 = st.columns(2)

with col1:
    units_2nd_approved = st.number_input("2nd semester units approved", min_value=0, max_value=26, value=5)
    units_1st_approved = st.number_input("1st semester units approved", min_value=0, max_value=26, value=5)
    units_2nd_enrolled = st.number_input("2nd semester units enrolled", min_value=0, max_value=26, value=6)
    grade_2nd = st.slider("2nd semester average grade (0-20 scale)", 0.0, 20.0, 12.0, 0.1)
    grade_1st = st.slider("1st semester average grade (0-20 scale)", 0.0, 20.0, 12.0, 0.1)
    age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=19)

with col2:
    tuition_up_to_date = st.selectbox("Tuition fees up to date?", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    debtor = st.selectbox("Is the student a debtor?", ["No", "Yes"])
    scholarship = st.selectbox("Scholarship holder?", ["No", "Yes"])
    course_code = st.number_input(
        "Course code (1-17, see dataset documentation)", min_value=1, max_value=17, value=1
    )
    unemployment_rate = st.number_input("Regional unemployment rate (%)", min_value=0.0, max_value=30.0, value=11.0)

# Convert human-friendly answers into the 0/1 codes the model expects
input_row = {
    "Curricular units 2nd sem (approved)": units_2nd_approved,
    "Tuition fees up to date": 1 if tuition_up_to_date == "Yes" else 0,
    "Curricular units 1st sem (approved)": units_1st_approved,
    "Curricular units 2nd sem (enrolled)": units_2nd_enrolled,
    "Course": course_code,
    "Age at enrollment": age,
    "Gender": 1 if gender == "Male" else 0,
    "Unemployment rate": unemployment_rate,
    "Curricular units 2nd sem (grade)": grade_2nd,
    "Curricular units 1st sem (grade)": grade_1st,
    "Debtor": 1 if debtor == "Yes" else 0,
    "Scholarship holder": 1 if scholarship == "Yes" else 0,
}

# ---- Predict button ----
if st.button("Check Risk", type="primary"):
    X_input = pd.DataFrame([input_row])[FEATURE_ORDER]

    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    predicted_label = TARGET_LABELS[prediction]

    st.subheader("Result")
    if predicted_label == "Dropout":
        st.error(f"⚠️ Predicted outcome: **{predicted_label}**")
    elif predicted_label == "Enrolled":
        st.warning(f"🟡 Predicted outcome: **{predicted_label}**")
    else:
        st.success(f"✅ Predicted outcome: **{predicted_label}**")

    # Show probability breakdown
    st.write("**Confidence breakdown:**")
    prob_df = pd.DataFrame({"Outcome": TARGET_LABELS, "Probability": probabilities})
    st.bar_chart(prob_df.set_index("Outcome"))

    # ---- Explain this specific prediction with SHAP ----
    st.subheader("Why this prediction?")
    with st.spinner("Explaining the prediction..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)

        # Explain the class the model actually predicted
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(shap_values[0, :, prediction], show=False)
        st.pyplot(fig)

    st.caption(
        "Bars show which details pushed the prediction toward the predicted outcome (positive) "
        "or away from it (negative). This model was trained on a public education dataset "
        "as a proof of concept and should not be used for real academic decisions without further validation."
    )
