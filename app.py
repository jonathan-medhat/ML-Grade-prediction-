import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===================== Page Config =====================
st.set_page_config(
    page_title="Student Grade & Scholarship Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Grade & Scholarship Predictor")
st.markdown("Predict your **academic grade** and explore the **best scholarships** available for you.")

# ===================== Load Resources =====================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("final_random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_scholarships():
    return pd.read_csv("ranked_scholarships.csv")

model, scaler = load_model_scaler()
sch_df = load_scholarships()

# ===================== User Input =====================
st.subheader("📊 Student Information")

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        weekly_study_hours = st.number_input(
            "Weekly Self Study Hours",
            min_value=0.0,
            max_value=168.0,
            value=20.0
        )

        attendance_percentage = st.number_input(
            "Attendance Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0
        )

    with col2:
        class_participation = st.slider(
            "Class Participation",
            min_value=0,
            max_value=10,
            value=5
        )

    submit = st.form_submit_button("Predict Grade & Scholarships")

# ===================== Prediction =====================
if submit:
    # -------- Prepare Input --------
    input_df = pd.DataFrame([{
        "weekly_self_study_hours": weekly_study_hours,
        "attendance_percentage": attendance_percentage,
        "class_participation": class_participation
    }])

    input_scaled = scaler.transform(input_df)

    # -------- Predict Grade --------
    grade_encoded = model.predict(input_scaled)[0]

    # Grade Mapping (Alphabetical – same as LabelEncoder)
    grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
    predicted_grade = grade_map.get(int(grade_encoded), "Unknown")

    st.success(f"🎯 Predicted Grade: **{predicted_grade}**")

    # ===================== Scholarships =====================
    st.subheader("🎓 Top Scholarships For You")

    # Filter by Grade
    filtered_sch = sch_df[
        sch_df["degrees"].astype(str).apply(
            lambda x: predicted_grade in [g.strip() for g in x.split(",")]
        )
    ].head(10)

    if filtered_sch.empty:
        st.warning("No scholarships found for this grade.")
    else:
        # Detect correct column names automatically
        cols = filtered_sch.columns.str.lower()

        name_col = next((c for c in filtered_sch.columns if "name" in c.lower()), None)
        country_col = next((c for c in filtered_sch.columns if "country" in c.lower()), None)
        funds_col = next((c for c in filtered_sch.columns if "fund" in c.lower()), None)
        score_col = "scholarship_score"

        display_cols = [c for c in [name_col, country_col, funds_col, score_col] if c is not None]

        display_df = filtered_sch[display_cols].copy()

        display_df.columns = [
            "Scholarship Name" if c == name_col else
            "Country" if c == country_col else
            "Funding Type" if c == funds_col else
            "Score"
            for c in display_cols
        ]

        st.dataframe(display_df, use_container_width=True)

# ===================== Footer =====================
st.markdown("---")
st.markdown("AI-Powered Academic & Scholarship Recommendation System")