import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Success Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# IMPORTANT:
# This CSS does NOT contain the page content.
# It only controls appearance.
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL PAGE
   ============================================================ */

.main {
    background-color: #f7f8fa;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-box {
    background: linear-gradient(
        135deg,
        #172554 0%,
        #1e3a8a 55%,
        #2563eb 100%
    );

    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.3rem;
    color: white;
}

.hero-title {
    color: white !important;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0;
}

.hero-subtitle {
    color: #e5edff !important;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-top: 0.7rem;
    margin-bottom: 0;
}


/* ============================================================
   STREAMLIT INPUTS
   ============================================================ */

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-weight: 600;
}


/* ============================================================
   BUTTON
   ============================================================ */

div.stButton > button {
    width: 100%;
    min-height: 3.1rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        padding: 1.35rem;
        border-radius: 14px;
    }

    .hero-title {
        font-size: 1.7rem;
    }

    .hero-subtitle {
        font-size: 0.92rem;
        line-height: 1.5;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CONSTANTS
# EXACTLY MATCH THE FINAL 12-FEATURE MODEL
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

TARGET_LABELS = [
    "Dropout",
    "Enrolled",
    "Graduate",
]


# ============================================================
# DATASET-BASED INPUT RANGES
# ============================================================

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
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except FileNotFoundError:

    st.error(
        "❌ The file 'dropout_model.pkl' could not be found. "
        "Make sure the model is in the same GitHub repository as app.py."
    )

    st.stop()

except Exception as exc:

    st.error(
        f"❌ The trained model could not be loaded.\n\n{exc}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Student Success")

    st.caption(
        "Machine-learning student outcome prediction"
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** XGBoost")
    st.write("**Features:** 12")
    st.write("**Classes:** 3")
    st.write("**Accuracy:** 74.01%")
    st.write("**Explainability:** SHAP")

    st.divider()

    st.markdown("### 🎯 Possible outcomes")

    st.write("🔴 **Dropout**")
    st.write("🟡 **Enrolled**")
    st.write("🟢 **Graduate**")

    st.divider()

    st.markdown("### ℹ️ About")

    st.caption(
        "This application is a data-science proof of concept "
        "built around the final 12-feature XGBoost model."
    )


# ============================================================
# HERO HEADER
# ============================================================
#
# IMPORTANT:
# The HTML tags below are intentionally NOT indented inside
# the HTML block. This prevents Streamlit from displaying them
# as literal text.
# ============================================================

st.markdown(
    """
<div class="hero-box">
<h1 class="hero-title">🎓 Student Success Intelligence</h1>
<p class="hero-subtitle">
Machine-learning based prediction of student academic outcomes:
Dropout, Enrolled, or Graduate.
</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# INTRODUCTION
# ============================================================

st.info(
    "Enter the student's academic, financial, demographic, and "
    "economic information below. The application uses the same "
    "12 features as the final deployed XGBoost model."
)


# ============================================================
# STUDENT PROFILE
# ============================================================

with st.container(border=True):

    st.markdown("### 👤 Student Profile")

    c1, c2 = st.columns(2)

    with c1:

        age = st.number_input(
            "Age at enrollment",
            min_value=AGE_MIN,
            max_value=AGE_MAX,
            value=19,
            step=1,
        )

        gender_label = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

    with c2:

        course = st.number_input(
            "Course ID",
            min_value=COURSE_MIN,
            max_value=COURSE_MAX,
            value=1,
            step=1,
            help=(
                "Course IDs 1–17 are the coded course "
                "values present in the dataset."
            ),
        )

        scholarship_label = st.selectbox(
            "Scholarship holder?",
            ["No", "Yes"],
        )


# ============================================================
# ACADEMIC PROGRESS
# ============================================================

with st.container(border=True):

    st.markdown("### 📚 Academic Progress")

    a1, a2 = st.columns(2)

    with a1:

        units_1st_approved = st.number_input(
            "1st semester units approved",
            min_value=0,
            max_value=SEM_1_APPROVED_MAX,
            value=5,
            step=1,
        )

        units_2nd_approved = st.number_input(
            "2nd semester units approved",
            min_value=0,
            max_value=SEM_2_APPROVED_MAX,
            value=5,
            step=1,
        )

        grade_1st = st.slider(
            "1st semester average grade",
            min_value=0.0,
            max_value=GRADE_MAX,
            value=12.0,
            step=0.1,
        )

    with a2:

        units_2nd_enrolled = st.number_input(
            "2nd semester units enrolled",
            min_value=0,
            max_value=SEM_2_ENROLLED_MAX,
            value=6,
            step=1,
        )

        grade_2nd = st.slider(
            "2nd semester average grade",
            min_value=0.0,
            max_value=GRADE_MAX,
            value=12.0,
            step=0.1,
        )


# ============================================================
# FINANCIAL & ECONOMIC FACTORS
# ============================================================

with st.container(border=True):

    st.markdown("### 💳 Financial & Economic Factors")

    f1, f2 = st.columns(2)

    with f1:

        tuition_label = st.selectbox(
            "Tuition fees up to date?",
            ["Yes", "No"],
        )

        debtor_label = st.selectbox(
            "Is the student a debtor?",
            ["No", "Yes"],
        )

    with f2:

        unemployment_rate = st.number_input(
            "Regional unemployment rate (%)",
            min_value=float(UNEMPLOYMENT_MIN),
            max_value=float(UNEMPLOYMENT_MAX),
            value=11.0,
            step=0.1,
        )


# ============================================================
# INPUT SUMMARY
# ============================================================

with st.expander("🔎 Review entered information"):

    review_data = pd.DataFrame(
        {
            "Feature": [
                "Age at enrollment",
                "Gender",
                "Course",
                "Scholarship holder",
                "1st semester units approved",
                "2nd semester units approved",
                "2nd semester units enrolled",
                "1st semester grade",
                "2nd semester grade",
                "Tuition fees up to date",
                "Debtor",
                "Unemployment rate",
            ],

            "Value": [
                age,
                gender_label,
                course,
                scholarship_label,
                units_1st_approved,
                units_2nd_approved,
                units_2nd_enrolled,
                f"{grade_1st:.1f}",
                f"{grade_2nd:.1f}",
                tuition_label,
                debtor_label,
                f"{unemployment_rate:.1f}%",
            ],
        }
    )

    st.dataframe(
        review_data,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# BUILD MODEL INPUT
# EXACT FEATURE ORDER
# ============================================================

input_row = {

    "Curricular units 2nd sem (approved)":
        units_2nd_approved,

    "Tuition fees up to date":
        1 if tuition_label == "Yes" else 0,

    "Curricular units 1st sem (approved)":
        units_1st_approved,

    "Curricular units 2nd sem (enrolled)":
        units_2nd_enrolled,

    "Course":
        course,

    "Age at enrollment":
        age,

    "Gender":
        1 if gender_label == "Male" else 0,

    "Unemployment rate":
        unemployment_rate,

    "Curricular units 2nd sem (grade)":
        grade_2nd,

    "Curricular units 1st sem (grade)":
        grade_1st,

    "Debtor":
        1 if debtor_label == "Yes" else 0,

    "Scholarship holder":
        1 if scholarship_label == "Yes" else 0,
}


X_input = (
    pd.DataFrame([input_row])
    [FEATURE_ORDER]
    .astype(float)
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("## 🔍 Analyze Student")

analyze = st.button(
    "Analyze Student",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    with st.spinner(
        "Analyzing the student profile..."
    ):

        try:

            prediction_raw = model.predict(
                X_input
            )[0]

            probabilities = model.predict_proba(
                X_input
            )[0]

            prediction = int(
                prediction_raw
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )

            st.stop()


    # ========================================================
    # MAP MODEL CLASS
    # ========================================================

    if prediction in [0, 1, 2]:

        predicted_label = (
            TARGET_LABELS[prediction]
        )

    else:

        st.error(
            "The model returned an unexpected class value."
        )

        st.stop()


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.markdown("## 📊 Prediction Result")


    if predicted_label == "Dropout":

        icon = "🔴"
        message = (
            "The model's highest-probability outcome "
            "is Dropout."
        )

    elif predicted_label == "Enrolled":

        icon = "🟡"
        message = (
            "The model's highest-probability outcome "
            "is Enrolled."
        )

    else:

        icon = "🟢"
        message = (
            "The model's highest-probability outcome "
            "is Graduate."
        )


    # ========================================================
    # MAIN RESULT CARD
    # ========================================================

    with st.container(border=True):

        st.markdown(
            "### Predicted outcome"
        )

        st.markdown(
            f"# {icon} {predicted_label}"
        )

        st.caption(
            message
        )


    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    st.markdown(
        "### 📈 Predicted Probabilities"
    )


    probability_df = pd.DataFrame(
        {
            "Outcome": TARGET_LABELS,
            "Probability": probabilities,
        }
    )


    # Probability cards

    p1, p2, p3 = st.columns(3)


    for col, label, value in zip(
        [p1, p2, p3],
        TARGET_LABELS,
        probabilities,
    ):

        with col:

            with st.container(border=True):

                if label == "Dropout":
                    emoji = "🔴"

                elif label == "Enrolled":
                    emoji = "🟡"

                else:
                    emoji = "🟢"

                st.metric(
                    label=f"{emoji} {label}",
                    value=f"{value:.1%}",
                )


    # ========================================================
    # BAR CHART
    # ========================================================

    st.markdown(
        "#### Outcome probability comparison"
    )


    chart_df = probability_df.copy()

    chart_df["Probability"] = (
        chart_df["Probability"] * 100
    )


    st.bar_chart(
        chart_df.set_index("Outcome")[
            "Probability"
        ],
        use_container_width=True,
    )


    st.caption(
        "These values represent the model's estimated "
        "probabilities for the three possible outcomes. "
        "They are not guarantees of future student behavior."
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.markdown(
        "## 🔎 Why did the model make this prediction?"
    )

    st.write(
        "SHAP explains how the student's 12 input features "
        "contributed to the model's prediction."
    )


    try:

        with st.spinner(
            "Generating SHAP explanation..."
        ):

            # ------------------------------------------------
            # Create SHAP explainer
            # ------------------------------------------------

            explainer = shap.TreeExplainer(
                model
            )

            shap_explanation = explainer(
                X_input
            )


            shap_array = np.asarray(
                shap_explanation.values
            )


            # ------------------------------------------------
            # Handle SHAP output dimensions
            # ------------------------------------------------

            if shap_array.ndim == 3:

                # Shape:
                # samples × features × classes

                class_values = (
                    shap_array[
                        0,
                        :,
                        prediction
                    ]
                )


            elif shap_array.ndim == 2:

                # Shape:
                # samples × features

                class_values = (
                    shap_array[0]
                )


            else:

                raise ValueError(
                    "Unexpected SHAP output shape: "
                    f"{shap_array.shape}"
                )


            # ------------------------------------------------
            # Create explanation dataframe
            # ------------------------------------------------

            explanation_df = pd.DataFrame(
                {
                    "Feature": FEATURE_ORDER,
                    "SHAP value": class_values,
                }
            )


            explanation_df[
                "Absolute impact"
            ] = (
                explanation_df[
                    "SHAP value"
                ].abs()
            )


            explanation_df = (
                explanation_df
                .sort_values(
                    "Absolute impact",
                    ascending=False,
                )
            )


            top_features = (
                explanation_df
                .head(8)
                .copy()
            )


            # ------------------------------------------------
            # Human-readable feature names
            # ------------------------------------------------

            readable_names = {

                "Curricular units 2nd sem (approved)":
                    "2nd sem units approved",

                "Tuition fees up to date":
                    "Tuition fees up to date",

                "Curricular units 1st sem (approved)":
                    "1st sem units approved",

                "Curricular units 2nd sem (enrolled)":
                    "2nd sem units enrolled",

                "Course":
                    "Course",

                "Age at enrollment":
                    "Age at enrollment",

                "Gender":
                    "Gender",

                "Unemployment rate":
                    "Unemployment rate",

                "Curricular units 2nd sem (grade)":
                    "2nd sem grade",

                "Curricular units 1st sem (grade)":
                    "1st sem grade",

                "Debtor":
                    "Debtor status",

                "Scholarship holder":
                    "Scholarship holder",
            }


            top_features[
                "Feature"
            ] = top_features[
                "Feature"
            ].map(
                readable_names
            )


            # ------------------------------------------------
            # SHAP chart
            # ------------------------------------------------

            st.markdown(
                "### Top factors influencing the prediction"
            )


            plot_df = (
                top_features
                .sort_values(
                    "SHAP value"
                )
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP value"],
            )


            ax.axvline(
                0,
                linewidth=1,
            )


            ax.set_xlabel(
                "SHAP value"
            )


            ax.set_title(
                f"Factors influencing "
                f"{predicted_label} prediction"
            )


            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True,
            )


            plt.close(fig)


            # ------------------------------------------------
            # SHAP interpretation
            # ------------------------------------------------

            st.markdown(
                "### 🧠 Top contributing factors"
            )


            for _, row in (
                top_features.iterrows()
            ):

                feature_name = row[
                    "Feature"
                ]

                shap_value = row[
                    "SHAP value"
                ]


                if shap_value > 0:

                    direction = (
                        "pushed the model toward "
                        f"the **{predicted_label}** prediction"
                    )

                elif shap_value < 0:

                    direction = (
                        "pushed the model away from "
                        f"the **{predicted_label}** prediction"
                    )

                else:

                    direction = (
                        "had approximately no directional "
                        "contribution"
                    )


                st.write(
                    f"**{feature_name}** — "
                    f"{direction} "
                    f"*(SHAP = {shap_value:.3f})*"
                )


            # ------------------------------------------------
            # SHAP explanation note
            # ------------------------------------------------

            st.caption(
                "A positive SHAP value means the feature "
                "contributed in the direction of the displayed "
                "predicted class for this explanation. "
                "A negative value means it contributed in the "
                "opposite direction."
            )


    except Exception as exc:

        st.warning(
            "The prediction was generated successfully, "
            "but the individual SHAP explanation could "
            f"not be displayed: {exc}"
        )


    # ========================================================
    # RESPONSIBLE USE
    # ========================================================

    st.divider()

    with st.container(border=True):

        st.markdown(
            "### ⚠️ Responsible Use"
        )

        st.warning(
            "This model is a proof-of-concept trained on "
            "a public education dataset. A prediction is a "
            "statistical estimate, not a definitive statement "
            "about a student's future. It should not be used "
            "as the sole basis for academic admissions, "
            "disciplinary action, financial decisions, or "
            "other high-impact decisions."
        )


# ============================================================
# ABOUT THE APPLICATION
# ============================================================

st.divider()


with st.expander(
    "📘 About this application"
):

    st.markdown(
        """
**Student Success Intelligence** is an interactive
machine-learning application built around the final
12-feature XGBoost model developed in the project notebook.

### Model

- **Algorithm:** XGBoost
- **Features:** 12
- **Outcomes:** Dropout, Enrolled, Graduate
- **Reported accuracy:** 74.01%
- **Explainability:** SHAP

### Model inputs

The application uses:

- 2nd semester units approved
- Tuition fees up to date
- 1st semester units approved
- 2nd semester units enrolled
- Course
- Age at enrollment
- Gender
- Regional unemployment rate
- 2nd semester grade
- 1st semester grade
- Debtor status
- Scholarship holder status

### Purpose

The application demonstrates how the trained model
can be used interactively and how individual predictions
can be interpreted using SHAP.

It is intended as a **data-science project demonstration**
rather than a production academic decision system.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🎓 Student Success Intelligence • "
    "XGBoost + SHAP • Data Science Project"
)
