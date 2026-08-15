import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---- Page setup ----
st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="centered")

# ---- Custom styling: education-themed background + hero banner ----
st.markdown(
    """
    <style>
    /* Force readable light text everywhere, on every widget, no exceptions */
    .stApp, .stApp * {
        color: #F5F5F5 !important;
    }

    .stApp {
        background-image:
            radial-gradient(circle at 8% 15%, rgba(245,166,35,0.06) 0%, transparent 8%),
            radial-gradient(circle at 92% 25%, rgba(245,166,35,0.05) 0%, transparent 10%),
            radial-gradient(circle at 15% 85%, rgba(245,166,35,0.05) 0%, transparent 9%),
            radial-gradient(circle at 88% 80%, rgba(245,166,35,0.06) 0%, transparent 8%);
        background-attachment: fixed;
    }

    /* Input boxes need their own lighter background so the white text is visible */
    input, textarea, select,
    div[data-baseweb="input"], div[data-baseweb="select"],
    div[data-baseweb="slider"] {
        background-color: #2A2E45 !important;
        color: #F5F5F5 !important;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1C1F2E 0%, #2A2E45 100%);
        border: 1px solid rgba(245,166,35,0.35);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .hero-banner .icons {
        font-size: 34px;
        letter-spacing: 18px;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .hero-banner h1 {
        color: #F5A623 !important;
        font-size: 30px;
        margin: 6px 0 4px 0;
    }
    .hero-banner p {
        color: #E5E5E5 !important;
        font-size: 15px;
        max-width: 560px;
        margin: 0 auto;
    }

    /* Style expander headers/bodies like cards, with guaranteed-readable text */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(245,166,35,0.25);
        border-radius: 12px;
        background-color: #161925 !important;
    }
    </style>

    <div class="hero-banner">
        <div class="icons">🎓 📘 ✏️ 📊 🏫</div>
        <h1>Student Dropout Risk Predictor</h1>
        <p>An AI tool estimating a student's risk of <b>dropping out</b>, <b>staying enrolled</b>,
        or <b>graduating</b> — and explaining exactly which factors drove that specific prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "📌 Just enter marks the normal way — as a **percentage (0-100%)**. The app converts it "
    "automatically behind the scenes, so no manual conversion needed.",
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

# Fixed, real defaults for the 2 fields that don't translate meaningfully across education
# systems (kept internally, not shown to the user) — computed from the actual training data.
DEFAULT_COURSE_CODE = 12       # most common course in the training data
DEFAULT_UNEMPLOYMENT_RATE = 11.57  # average unemployment rate in the training data

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
        percent_1st = st.slider(
            "Average marks (%)",
            0, 100, 60, 1,
            help="Average percentage across 1st-semester subjects — enter it the normal way, no conversion needed.",
            key="p1",
        )
    with c2:
        st.caption("2nd Semester")
        units_2nd_approved = st.number_input(
            "Subjects passed",
            min_value=0, max_value=26, value=5,
            help="How many 2nd-semester subjects did the student pass (not just attempt)?",
            key="u2a",
        )
        percent_2nd = st.slider(
            "Average marks (%)",
            0, 100, 60, 1,
            help="Average percentage across 2nd-semester subjects — enter it the normal way, no conversion needed.",
            key="p2",
        )

    # Convert percentage -> the model's internal 0-20 scale, invisibly
    grade_1st = (percent_1st / 100) * 20
    grade_2nd = (percent_2nd / 100) * 20

    units_2nd_enrolled = st.number_input(
        "Subjects registered for in 2nd semester (whether passed or not)",
        min_value=0, max_value=26, value=6,
        help="Total number of subjects the student signed up for in the 2nd semester, including any they didn't pass.",
    )

with st.expander("💰 Financial Details", expanded=True):
    tuition_up_to_date = st.radio(
        "Is the student up to date on tuition payments?",
        ["Yes", "No"],
        horizontal=True,
    )
    debtor = st.radio(
        "Does the student owe any other outstanding fees to the institution?",
        ["No", "Yes"],
        horizontal=True,
    )
    scholarship = st.radio(
        "Is the student on a scholarship?",
        ["No", "Yes"],
        horizontal=True,
    )

with st.expander("👤 Personal Details", expanded=True):
    age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=19)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

# Build the row the model expects — 2 fields (Course, Unemployment rate) use fixed real
# defaults from the training data instead of asking the user, since they don't translate
# meaningfully across education systems and have a smaller effect on the prediction anyway.
input_row = {
    "Curricular units 2nd sem (approved)": units_2nd_approved,
    "Tuition fees up to date": 1 if tuition_up_to_date == "Yes" else 0,
    "Curricular units 1st sem (approved)": units_1st_approved,
    "Curricular units 2nd sem (enrolled)": units_2nd_enrolled,
    "Course": DEFAULT_COURSE_CODE,
    "Age at enrollment": age,
    "Gender": 1 if gender == "Male" else 0,
    "Unemployment rate": DEFAULT_UNEMPLOYMENT_RATE,
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
