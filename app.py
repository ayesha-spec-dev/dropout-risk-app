import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---- Page setup ----
st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Dropout Risk Predictor")
st.markdown(
    "This AI tool estimates a student's risk of **dropping out**, **staying enrolled**, "
    "or **graduating**, based on their academic performance and background — and explains "
    "which factors matter most for that specific prediction."
)

st.info(
    "📌 **Note:** This model was trained on a European university dataset, which uses a "
    "**0-20 grading scale** (not percentages or GPA). Rough conversion: a score of 20 = "
    "excellent (like 90%+), 14-16 = good (like 70-80%), 10 = just passing (like 50%), "
    "below 10 = failing. Use your best estimate if converting from a percentage system.",
    icon="ℹ️",
)

# ---- Load the trained model ----
model = joblib.load("dropout_model.pkl")

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

st.divider()
st.subheader("📋 Student Details")

with st.expander("📚 Academic Performance", expanded=True):
    st.caption("How many subjects the student registered for vs. actually passed, each semester.")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("1st Semester")
        units_1st_approved = st.number_input(
            "Subjects passed",
            min_value=0, max_value=26, value=5,
            help="How many 1st-semester subjects did the student pass (not just attempt)?",
            key="u1a",
        )
        grade_1st = st.slider(
            "Average grade (0-20 scale)",
            0.0, 20.0, 12.0, 0.1,
            help="Average score across 1st-semester subjects, on the 0-20 scale explained above.",
            key="g1",
        )
    with c2:
        st.caption("2nd Semester")
        units_2nd_approved = st.number_input(
            "Subjects passed",
            min_value=0, max_value=26, value=5,
            help="How many 2nd-semester subjects did the student pass (not just attempt)?",
            key="u2a",
        )
        grade_2nd = st.slider(
            "Average grade (0-20 scale)",
            0.0, 20.0, 12.0, 0.1,
            help="Average score across 2nd-semester subjects, on the 0-20 scale explained above.",
            key="g2",
        )

    units_2nd_enrolled = st.number_input(
        "Subjects registered for in 2nd semester (enrolled, whether passed or not)",
        min_value=0, max_value=26, value=6,
        help="Total number of subjects the student signed up for in the 2nd semester, including any they didn't pass.",
    )

with st.expander("💰 Financial & Background Details", expanded=True):
    tuition_up_to_date = st.radio(
        "Is the student up to date on tuition payments?",
        ["Yes", "No"],
        horizontal=True,
        help="Has the student paid their tuition fees on schedule, with nothing overdue?",
    )
    debtor = st.radio(
        "Does the student owe any other outstanding debt to the university?",
        ["No", "Yes"],
        horizontal=True,
        help="Separate from tuition — any other unpaid fees or debts owed to the institution.",
    )
    scholarship = st.radio(
        "Is the student on a scholarship?",
        ["No", "Yes"],
        horizontal=True,
    )
    unemployment_rate = st.slider(
        "Local unemployment rate at the time (%)",
        0.0, 30.0, 11.0, 0.5,
        help="The general unemployment rate in the student's region/country around this time. "
             "If unsure, a typical value (10-12%) is a reasonable default.",
    )

with st.expander("👤 Personal Details", expanded=True):
    age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=19)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    course_code = st.number_input(
        "Course/program code",
        min_value=1, max_value=17, value=1,
        help="A numeric code (1-17) representing the specific degree program, as coded in the "
             "original dataset. If you don't know the exact code, leave the default — it has a "
             "smaller effect on the prediction than the academic and financial fields above.",
    )

# Build the row the model expects
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

st.divider()

if st.button("🔍 Check Risk", type="primary", use_container_width=True):
    X_input = pd.DataFrame([input_row])[FEATURE_ORDER]

    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    predicted_label = TARGET_LABELS[prediction]

    st.subheader("📊 Result")

    if predicted_label == "Dropout":
        st.error(f"### ⚠️ Predicted outcome: {predicted_label}")
        st.write("This student profile shows a **higher risk** of dropping out based on similar historical cases.")
    elif predicted_label == "Enrolled":
        st.warning(f"### 🟡 Predicted outcome: {predicted_label}")
        st.write("This student's outcome is **uncertain** — worth monitoring, not yet clearly headed toward either dropout or graduation.")
    else:
        st.success(f"### ✅ Predicted outcome: {predicted_label}")
        st.write("This student profile shows a **strong likelihood of graduating** based on similar historical cases.")

    st.write("**Confidence breakdown:**")
    prob_df = pd.DataFrame({"Outcome": TARGET_LABELS, "Probability": probabilities})
    st.bar_chart(prob_df.set_index("Outcome"), color="#7C3AED")

    st.subheader("🔎 Why this prediction?")
    with st.spinner("Explaining the prediction..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(shap_values[0, :, prediction], show=False)
        st.pyplot(fig)

    st.caption(
        "Bars show which details pushed the prediction toward the predicted outcome (positive) "
        "or away from it (negative). Built on a public education dataset as an academic proof "
        "of concept — not intended for real academic decision-making without further validation."
    )
