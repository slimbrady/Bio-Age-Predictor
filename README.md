# 🧬 Biological Age & Longevity Predictor

This professional health-tech application predicts a user's **Biological Age** and **Mortality Risk** using machine learning models (XGBoost, ANN) trained on the **NHANES (National Health and Nutrition Examination Survey)** dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://streamlit.io](https://realagepredictor.streamlit.app/))


## 🚀 Features
- **Predictive Modeling**: Uses XGBoost to estimate biological age based on blood markers, vitals, and physical activity.
- **Smart Data Handling**: 
    - Automatic conversion (Weight lbs -> kg, Height ft/in -> cm, Waist in -> cm).
    - Multi-step input UI for precision tracking.
- **Profile Persistence**: Save and load user profiles locally using JSON snapshots (Initial/Birth Year based).
- **Comprehensive Markers**: Analyzes markers like CRP, Albumin, Glucose, LDL/HDL, VO2 Max, and more.

## 🛠️ Tech Stack
- **Python 3.11**
- **Streamlit** (UI/UX)
- **XGBoost** (Machine Learning)
- **Scikit-learn** (Data Scaling)
- **Docker** (Containerization)

## 📦 How to Run
1. Clone this repository.
2. Ensure you have the `.joblib` and `.pkl` asset files in the directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`



---
*Created as part of a Longevity Research project using NHANES public health data.*
