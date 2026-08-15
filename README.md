# 🎓 Student Success Intelligence

An XGBoost-based machine learning project for predicting student outcomes.

## 🚀 Live Demo

👉 **[Launch Student Success Intelligence App](https://dropout-risk-app-6aqozrjwi3gbymruek7dba.streamlit.app/)**

## 📌 Project Overview

...

---

## Overview

Student dropout is a costly, high-stakes problem for universities everywhere — lost tuition, wasted potential, and students who needed support they never got. This project builds an early-warning system: a machine learning model that flags at-risk students using only the information a university already has at enrollment and after the first semester.

**Research question:** Can we predict a student's academic outcome (Dropout / Enrolled / Graduate) from their academic, financial, and demographic profile — and do the risk factors differ between male and female students?

## Dataset

- **Source:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — UCI Machine Learning Repository, created by researchers at the Polytechnic Institute of Portalegre, Portugal (Realinho et al., 2021, CC BY 4.0 license)
- **4,424 real student records**, 35 features spanning academic path, demographics, and socioeconomic background
- **No missing values, no duplicates** — genuinely clean research data
- **Note:** this dataset is Portuguese, not Pakistani. It was chosen as a proof-of-concept because it is a well-documented, high-quality, individual-level dataset with a real gender field — ideal for learning and demonstrating the full technique. The same modeling approach is directly transferable to Pakistani education data (e.g. the Punjab MICS household survey), which is a natural next step for this project.

## Key Findings

### Model performance

| Model | Accuracy | Enrolled Recall |
|---|---|---|
| Random Forest | 78.1% | 38% |
| Logistic Regression (no weighting) | 75.5% | 32% |
| **XGBoost (final model)** | **75.5%** | **47%** |
| Logistic Regression (balanced) | 74.0% | 65% |

The plain baseline looked strong on accuracy alone, but poorly identified "Enrolled" students — the smallest and most ambiguous outcome group, and arguably the one an early-warning system most needs to catch. **XGBoost was selected as the final model** for its balance of overall accuracy and meaningfully better recall on this hard-to-predict group.

### Gender and dropout risk

Male students in this dataset drop out at nearly **double the rate** of female students (45% vs 25%). Using SHAP explainability, we found that **the underlying drivers are almost identical for both genders** — semester units passed and tuition payment status dominate for both. Gender itself is a secondary factor, not a primary driver. Two smaller nuances: enrollment age matters more for male students' risk, while grades matter slightly more for female students'.

**Conclusion:** the dropout gap is not explained by gender-specific causes — both groups are driven by the same academic and financial pressures, which simply appear to affect male students more severely here. This suggests interventions (tuition support, early academic monitoring) would likely help both groups, rather than needing separate strategies by gender.

## The App

A simplified version of the model (retrained on its 12 most important features, for a usable form) powers a live Streamlit app: enter a student's academic and financial details, get an instant risk prediction, and see a chart explaining exactly which factors drove that specific prediction — powered by the same SHAP technique used in the analysis.

## Repository Structure

```
dropout-risk-prediction/
├── Dropout_Project.ipynb   # Full analysis: EDA, modeling, SHAP explainability
├── dataset.csv               # Source data (UCI/Kaggle)
├── app.py                    # Streamlit deployment app
├── requirements.txt          # App dependencies
├── dropout_model.pkl         # Simplified model powering the app
└── README.md
```

## How to Run

1. Clone this repo.
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit`
3. Open `Dropout_Project.ipynb` to see the full analysis, or run `streamlit run app.py` to launch the app locally.

## Limitations

- Dataset is Portuguese higher-education data, not Pakistani — findings are a proof of concept, not a direct statement about Pakistani students.
- The deployed app uses a simplified 12-feature model for usability, which trades some accuracy for a manageable input form.
- This is an academic project, not a validated tool for real academic decision-making.

## Future Work

- Apply the same pipeline to Pakistani education survey data (e.g. Punjab MICS)
- Expand the app's feature set for higher accuracy
- Add batch prediction for entire class cohorts

## Author

**Ayesha Ghani** — BS Data Science, Air University, Islamabad
