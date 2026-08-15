import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Success Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

<style>
/* ============================================================
   GLOBAL APP STYLING
   ============================================================ */

.main {
    background-color: #f7f8fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}


/* ============================================================
   HERO HEADER
   ============================================================ */

.hero {
    padding: 2rem 2.2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #172554 0%,
        #1e3a8a 55%,
        #2563eb 100%
    );
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 750;
    color: white !important;
}

.hero p {
    margin: 0.55rem 0 0;
    font-size: 1.05rem;
    opacity: 0.9;
    color: white !important;
}


/* ============================================================
   INPUT SECTION CARDS
   ============================================================ */

.section-card {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.25rem 1.4rem 0.8rem;
    margin-bottom: 1rem;
}


/* ============================================================
   SECTION TITLES
   IMPORTANT: Explicit dark color prevents mobile theme
   from making the headings white on the white card.
   ============================================================ */

.section-title {
    color: #111827 !important;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.85rem;
}


/* ============================================================
   RESULT CARD
   ============================================================ */

.result-card {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    min-height: 170px;
}

.result-label {
    color: #6b7280 !important;
    font-size: 0.9rem;
    margin-bottom: 0.35rem;
}

.result-value {
    color: #111827 !important;
    font-size: 2rem;
    font-weight: 800;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    color: #111827 !important;
    font-size: 1.55rem;
    font-weight: 750;
}

.metric-label {
    color: #6b7280 !important;
    font-size: 0.85rem;
}

.small-note {
    color: #6b7280 !important;
    font-size: 0.82rem;
}


/* ============================================================
   ANALYZE BUTTON
   ============================================================ */

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3rem;
    font-weight: 700;
}


/* ============================================================
   MOBILE SAFETY
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 1.4rem;
        border-radius: 14px;
    }

    .hero h1 {
        font-size: 1.8rem;
    }

    .hero p {
        font-size: 0.95rem;
    }

    .section-card {
        background-color: #ffffff !important;
        color: #111827 !important;
        padding: 1rem;
        border-radius: 14px;
    }

    .section-title {
        color: #111827 !important;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .result-card,
    .metric-card {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    .result-label,
    .metric-label,
    .small-note {
        color: #6b7280 !important;
    }

    .result-value,
    .metric-value {
        color: #111827 !important;
    }
}
</style>

# ============================================================
# CONSTANTS — MATCH THE TRAINED MODEL EXACTLY
# ============================================================
MODEL_PATH = "dropout_model.pkl"

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

# Ranges are based on the project dataset.
SEM_2_APPROVED_MAX = 20
SEM_1_APPROVED_MAX = 26
SEM_2_ENROLLED_MAX = 23
GRADE_MAX = 20.0
COURSE_MIN = 1
COURSE_MAX = 17
AGE_MIN = 17
AGE_MAX = 70
UNEMPLOYMENT_MIN = 7.6
UNEMPLOYMENT_MAX = 16.2

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "The model file 'dropout_model.pkl' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 Student Success")
    st.caption("Machine-learning student outcome prediction")

    st.divider()

    st.markdown("### Model information")
    st.write("**Model:** XGBoost")
    st.write("**Inputs:** 12 features")
    st.write("**Outcomes:** 3 classes")
    st.write("**Reported accuracy:** 74.01%")
    st.write("**Explainability:** SHAP")

    st.divider()

    st.markdown("### Outcomes")
    st.write("🔴 **Dropout**")
    st.write("🟡 **Enrolled**")
    st.write("🟢 **Graduate**")

    st.divider()

    st.caption(
        "This is a proof-of-concept machine-learning application. "
        "Predictions should not be used as the sole basis for academic decisions."
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Success Intelligence</h1>
        <p>
            Machine-learning based prediction of student academic outcomes:
            Dropout, Enrolled, or Graduate.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Enter the student's academic, financial, demographic, and economic information "
    "below. The application uses the same 12 features as the deployed XGBoost model."
)

# ============================================================
# INPUTS
# ============================================================
st.markdown(
    '<div class="section-card"><div class="section-title">👤 Student Profile</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    age = st.number_input(
        "Age at enrollment",
        min_value=AGE_MIN,
        max_value=AGE_MAX,
        value=19,
        step=1,
    )

with c2:
    gender_label = st.selectbox("Gender", ["Female", "Male"])

with c3:
    course = st.number_input(
        "Course ID",
        min_value=COURSE_MIN,
        max_value=COURSE_MAX,
        value=1,
        step=1,
        help="Course IDs 1–17 are the coded course values present in the dataset.",
    )

with c4:
    scholarship_label = st.selectbox(
        "Scholarship holder?",
        ["No", "Yes"],
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-card"><div class="section-title">📚 Academic Progress</div>',
    unsafe_allow_html=True,
)

a1, a2, a3 = st.columns(3)

with a1:
    units_1st_approved = st.number_input(
        "1st semester units approved",
        min_value=0,
        max_value=SEM_1_APPROVED_MAX,
        value=5,
        step=1,
    )

with a2:
    units_2nd_approved = st.number_input(
        "2nd semester units approved",
        min_value=0,
        max_value=SEM_2_APPROVED_MAX,
        value=5,
        step=1,
    )

with a3:
    units_2nd_enrolled = st.number_input(
        "2nd semester units enrolled",
        min_value=0,
        max_value=SEM_2_ENROLLED_MAX,
        value=6,
        step=1,
    )

a4, a5 = st.columns(2)

with a4:
    grade_1st = st.slider(
        "1st semester average grade",
        min_value=0.0,
        max_value=GRADE_MAX,
        value=12.0,
        step=0.1,
    )

with a5:
    grade_2nd = st.slider(
        "2nd semester average grade",
        min_value=0.0,
        max_value=GRADE_MAX,
        value=12.0,
        step=0.1,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-card"><div class="section-title">💳 Financial & Economic Factors</div>',
    unsafe_allow_html=True,
)

f1, f2, f3 = st.columns(3)

with f1:
    tuition_label = st.selectbox(
        "Tuition fees up to date?",
        ["Yes", "No"],
    )

with f2:
    debtor_label = st.selectbox(
        "Is the student a debtor?",
        ["No", "Yes"],
    )

with f3:
    unemployment_rate = st.number_input(
        "Regional unemployment rate (%)",
        min_value=float(UNEMPLOYMENT_MIN),
        max_value=float(UNEMPLOYMENT_MAX),
        value=11.0,
        step=0.1,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# BUILD MODEL INPUT
# ============================================================
input_row = {
    "Curricular units 2nd sem (approved)": units_2nd_approved,
    "Tuition fees up to date": 1 if tuition_label == "Yes" else 0,
    "Curricular units 1st sem (approved)": units_1st_approved,
    "Curricular units 2nd sem (enrolled)": units_2nd_enrolled,
    "Course": course,
    "Age at enrollment": age,
    "Gender": 1 if gender_label == "Male" else 0,
    "Unemployment rate": unemployment_rate,
    "Curricular units 2nd sem (grade)": grade_2nd,
    "Curricular units 1st sem (grade)": grade_1st,
    "Debtor": 1 if debtor_label == "Yes" else 0,
    "Scholarship holder": 1 if scholarship_label == "Yes" else 0,
}

X_input = pd.DataFrame([input_row])[FEATURE_ORDER].astype(float)

# ============================================================
# PREDICTION
# ============================================================
st.markdown("### 🔍 Analyze Student")

if st.button("Analyze Student", type="primary"):
    with st.spinner("Analyzing student profile..."):
        try:
            prediction = int(model.predict(X_input)[0])
            probabilities = model.predict_proba(X_input)[0]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    predicted_label = TARGET_LABELS[prediction]

    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------
    st.markdown("## Prediction")

    result_col, prob_col = st.columns([1, 2])

    with result_col:
        if predicted_label == "Dropout":
            icon = "🔴"
        elif predicted_label == "Enrolled":
            icon = "🟡"
        else:
            icon = "🟢"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted outcome</div>
                <div class="result-value">{icon} {predicted_label}</div>
                <div class="small-note">Highest model probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with prob_col:
        st.markdown("#### Predicted probabilities")

        probability_df = pd.DataFrame(
            {
                "Outcome": TARGET_LABELS,
                "Probability": probabilities * 100,
            }
        )

        st.bar_chart(
            probability_df.set_index("Outcome"),
            y="Probability",
        )

        st.caption(
            "These are model probabilities, not guarantees of a student's future outcome."
        )

    # --------------------------------------------------------
    # Probability cards
    # --------------------------------------------------------
    st.markdown("#### Outcome breakdown")

    p1, p2, p3 = st.columns(3)

    for col, label, value in zip(
        [p1, p2, p3],
        TARGET_LABELS,
        probabilities,
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------
    st.markdown("## 🔎 Why did the model make this prediction?")
    st.caption(
        "SHAP shows how the student's input values influenced the model's prediction. "
        "The explanation is generated from the same XGBoost model used for prediction."
    )

    try:
        with st.spinner("Generating explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)

            # XGBoost SHAP output can be either:
            # (samples, features, classes) or a list of class arrays.
            if isinstance(shap_values, list):
                class_values = np.asarray(shap_values[prediction])[0]
            else:
                shap_array = np.asarray(shap_values)

                if shap_array.ndim == 3:
                    class_values = shap_array[0, :, prediction]
                elif shap_array.ndim == 2:
                    class_values = shap_array[0]
                else:
                    raise ValueError(
                        f"Unexpected SHAP output shape: {shap_array.shape}"
                    )

            explanation_df = pd.DataFrame(
                {
                    "Feature": FEATURE_ORDER,
                    "SHAP value": class_values,
                }
            )

            explanation_df["Absolute impact"] = explanation_df["SHAP value"].abs()
            explanation_df = explanation_df.sort_values(
                "Absolute impact",
                ascending=False,
            )

            top_features = explanation_df.head(8).copy()

            # Human-readable labels.
            readable_names = {
                "Curricular units 2nd sem (approved)": "2nd sem units approved",
                "Tuition fees up to date": "Tuition fees up to date",
                "Curricular units 1st sem (approved)": "1st sem units approved",
                "Curricular units 2nd sem (enrolled)": "2nd sem units enrolled",
                "Course": "Course",
                "Age at enrollment": "Age at enrollment",
                "Gender": "Gender",
                "Unemployment rate": "Unemployment rate",
                "Curricular units 2nd sem (grade)": "2nd sem grade",
                "Curricular units 1st sem (grade)": "1st sem grade",
                "Debtor": "Debtor status",
                "Scholarship holder": "Scholarship holder",
            }

            top_features["Feature"] = top_features["Feature"].map(readable_names)

            fig, ax = plt.subplots(figsize=(9, 4.8))

            plot_df = top_features.sort_values("SHAP value")
            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP value"],
            )

            ax.axvline(0, linewidth=1)
            ax.set_xlabel("SHAP value")
            ax.set_title(f"Top factors influencing: {predicted_label}")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("#### Top contributing factors")

            for _, row in top_features.iterrows():
                direction = (
                    "toward"
                    if row["SHAP value"] > 0
                    else "away from"
                )

                st.write(
                    f"**{row['Feature']}** — "
                    f"{direction} the `{predicted_label}` prediction "
                    f"(SHAP: {row['SHAP value']:.3f})"
                )

    except Exception as exc:
        st.warning(
            "The prediction was generated successfully, but the individual "
            f"SHAP explanation could not be displayed: {exc}"
        )

    st.divider()

    # --------------------------------------------------------
    # Responsible use
    # --------------------------------------------------------
    st.markdown("### ⚠️ Responsible use")

    st.warning(
        "This model is a proof-of-concept trained on a public student dataset. "
        "A prediction is a statistical estimate, not a definitive statement about "
        "a student's future. It should not be used as the sole basis for academic "
        "admissions, disciplinary action, financial decisions, or other high-impact "
        "decisions."
    )

# ============================================================
# FOOTER / PROJECT DETAILS
# ============================================================
st.divider()

with st.expander("About this application"):
    st.markdown(
        """
        **Student Success Intelligence** is an interactive application built around
        the final 12-feature XGBoost model developed in the project notebook.

        **Model outputs**
        - Dropout
        - Enrolled
        - Graduate

        **Model performance**
        - Reported accuracy: **74.01%**

        **Model inputs**
        - Academic progress
        - Tuition/payment status
        - Student demographics
        - Course
        - Regional unemployment rate

        **Explainability**
        - Individual predictions can be examined using SHAP feature contributions.

        The application is intended as a data-science proof of concept rather than
        a production academic decision system.
        """
    )

st.caption(
    "Student Success Intelligence • XGBoost + SHAP • Data Science Project"
)
