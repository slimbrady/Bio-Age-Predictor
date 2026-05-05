import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bio-Age Predictor", page_icon="🧬", layout="wide")

# Custom CSS for Realistic Buttons and Beautiful Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Lexend:wght@400;600&display=swap');

    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.88)), 
                    url("https://images.unsplash.com/photo-1541534741688-6078c64b5903?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #1a1a1a !important;
    }

    h1 {
        font-family: 'Lexend', sans-serif !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem !important;
        text-align: center;
    }

    h2, h3, [data-testid="stExpander"] label {
        font-family: 'Lexend', sans-serif !important;
        font-weight: 600 !important;
        color: #1e3a8a !important;
    }

    /* Blue Slider Accent */
    div[data-testid="stSlider"] [data-testid="stThumb"] {
        background-color: #3b82f6 !important;
        border: 2px solid #1d4ed8 !important;
    }
    div[data-testid="stSlider"] [data-testid="stTrack"] {
        background-color: #dbeafe !important;
    }

    /* Uniform Keyboard-style Buttons */
    div.stButton > button {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0) !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 5px 0 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 0 #94a3b8, 
                    0 5px 10px rgba(0,0,0,0.05) !important;
        transition: all 0.05s ease !important;
        width: 100% !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(145deg, #e2e8f0, #cbd5e1) !important;
        box-shadow: 0 2px 0 #64748b, 0 3px 6px rgba(0,0,0,0.08) !important;
        transform: translateY(2px) !important;
    }
    
    div.stButton > button:active {
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
        transform: translateY(4px) !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(145deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border: 1px solid #1d4ed8 !important;
        box-shadow: 0 4px 0 #1e3a8a, 0 5px 10px rgba(0,0,0,0.2) !important;
    }

    .shaded-input {
        opacity: 0.4;
        filter: grayscale(100%);
        pointer-events: none;
        background: #f1f5f9;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }

    [data-testid="stExpander"], [data-testid="column"], [data-testid="stVerticalBlock"] > div:has(div.stExpander) {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid #f1f5f9;
    }

    .recommendation-card {
        background: #f0f9ff;
        border-left: 5px solid #0ea5e9;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px 8px 8px 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def pace_to_mph(p_str):
    try:
        m, s = map(int, p_str.split(':'))
        total_mins = m + (s / 60.0)
        return 60.0 / total_mins
    except: return 6.67 # 9:00 default

def get_user_file(initials, birth_year):
    if not initials or len(str(initials)) < 3: return None
    return os.path.join(os.path.dirname(__file__), f"profile_{str(initials).lower()}_{birth_year}.json")

def save_profile(data, initials, birth_year):
    path = get_user_file(initials, birth_year)
    if path:
        try:
            with open(path, "w") as f: json.dump(data, f)
        except: pass

def load_profile(initials, birth_year):
    path = get_user_file(initials, birth_year)
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: return None
    return None

@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_dir, "xgb_bioage_model.joblib"))
        scaler = joblib.load(os.path.join(base_dir, "scaler_x.pkl"))
        features = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
        return model, scaler, features
    except: return None, None, None

# --- STATE ---
DEFAULT_STATE = {
    'age': 45.0, 'weight': 145.0, 'h_ft': 5, 'h_in': 10,
    'waist_in': 34.0, 'pct_bft': 18.0, 'sys': 115.0, 'dia': 75.0, 'pulse': 60.0,
    'vo2': 42.0, 'crp': 0.1, 'trig': 80.0, 'ldl': 90.0, 'hdl': 60.0, 'gluc': 88.0, 'alb': 4.5, 'iron': 100.0,
    's1': "Walking", 'd1': 5, 'i1': "Moderate",
    's2': "Running", 'd2': 0, 'i2': "Vigorous",
    's3': "None", 'd3': 0, 'i3': "Moderate",
    'has_crp': False, 'has_trig': False, 'has_ldl': False, 'has_hdl': False,
    'has_gluc': False, 'has_alb': False, 'has_iron': False,
    'walk_speed': 3.1, 'run_pace': "9:00"
}
MARKER_DEFAULTS = {'trig': 80.0, 'ldl': 90.0, 'hdl': 60.0, 'gluc': 88.0, 'crp': 0.1, 'alb': 4.5, 'iron': 100.0}

if 'profile_data' not in st.session_state: st.session_state.profile_data = DEFAULT_STATE.copy()
if 'user_loaded' not in st.session_state: st.session_state.user_loaded = False
if 'cur_initials' not in st.session_state: st.session_state.cur_initials = ""
if 'cur_birth_year' not in st.session_state: st.session_state.cur_birth_year = 1980

st.markdown("<h1>🧬 Biological Age Predictor</h1>", unsafe_allow_html=True)

with st.expander("👤 User Profile", expanded=not st.session_state.user_loaded):
    col_init, col_year = st.columns(2)
    i_in = col_init.text_input("Initials (3)", st.session_state.cur_initials, max_chars=3).upper()
    y_in = col_year.number_input("Birth Year", 1920, 2024, st.session_state.cur_birth_year)
    if st.button("Load Profile"):
        if len(i_in) == 3:
            st.session_state.cur_initials, st.session_state.cur_birth_year = i_in, y_in
            data = load_profile(i_in, y_in)
            if data: st.session_state.profile_data.update(data)
            st.session_state.user_loaded = True
            st.rerun()

def multi_step_input(label, key, min_val, max_val, step_small=0.5, step_large=5.0, enabled=True):
    if key not in st.session_state: st.session_state[key] = float(st.session_state.profile_data.get(key, min_val))
    if not enabled:
        st.markdown(f"<div class='shaded-input'><strong>{label} (Auto)</strong><br><small>Clinical: {MARKER_DEFAULTS.get(key, 'N/A')}</small></div>", unsafe_allow_html=True)
        return float(MARKER_DEFAULTS.get(key, min_val))
    st.write(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button(f"-{int(step_large)}", key=f"b1_{key}"): st.session_state[key] = max(min_val, st.session_state[key]-step_large); st.rerun()
    if c2.button(f"-{step_small}", key=f"b2_{key}"): st.session_state[key] = max(min_val, st.session_state[key]-step_small); st.rerun()
    if c3.button(f"+{step_small}", key=f"b3_{key}"): st.session_state[key] = min(max_val, st.session_state[key]+step_small); st.rerun()
    if c4.button(f"+{int(step_large)}", key=f"b4_{key}"): st.session_state[key] = min(max_val, st.session_state[key]+step_large); st.rerun()
    return st.number_input(label, min_val, max_val, step=1.0, key=key, label_visibility="collapsed")

with st.sidebar:
    st.markdown("### 📋 Vitals")
    chrono_age = st.number_input("Age", 18, 100, int(st.session_state.profile_data.get('age', 45)))
    weight_lb = multi_step_input("Weight (lbs)", "weight", 80.0, 500.0, 1.0, 5.0)
    h_ft = st.selectbox("Feet", range(4, 9), index=int(st.session_state.profile_data.get('h_ft', 5)-4))
    h_in = st.selectbox("Inches", range(12), index=int(st.session_state.profile_data.get('h_in', 10)))
    waist_in = multi_step_input("Waist (in)", "waist_in", 20.0, 70.0, 0.5, 5.0)
    pct_bft = multi_step_input("Body Fat %", "pct_bft", 3.0, 60.0, 0.5, 5.0)
    sys_bp = multi_step_input("Systolic BP", "sys", 80.0, 200.0, 1.0, 5.0)
    dia_bp = multi_step_input("Diastolic BP", "dia", 40.0, 120.0, 1.0, 5.0)
    pulse = multi_step_input("Pulse", "pulse", 40.0, 150.0, 1.0, 5.0)
    st.divider()
    st.markdown("### 🩺 Labs")
    def lab(l, k, mk, mi, ma, ss, sl):
        h = st.checkbox(f"Have {l}?", st.session_state.profile_data.get(mk, False), key=mk)
        return multi_step_input(l, k, mi, ma, ss, sl, enabled=h), h
    v_crp, h_crp = lab("CRP", "crp", "has_crp", 0.0, 15.0, 0.1, 1.0)
    v_trig, h_trig = lab("Trig", "trig", "has_trig", 0.0, 500.0, 1.0, 10.0)
    v_ldl, h_ldl = lab("LDL", "ldl", "has_ldl", 0.0, 350.0, 1.0, 10.0)
    v_hdl, h_hdl = lab("HDL", "hdl", "has_hdl", 0.0, 150.0, 1.0, 10.0)
    v_gluc, h_gluc = lab("Glucose", "gluc", "has_gluc", 0.0, 300.0, 1.0, 10.0)
    v_alb, h_alb = lab("Albumin", "alb", "has_alb", 0.0, 6.0, 0.1, 0.5)
    v_iron, h_iron = lab("Iron", "iron", "has_iron", 0.0, 300.0, 1.0, 10.0)

st.subheader("🏆 Fitness & Mobility")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    v_walk = st.slider("Walking Speed (mph)", 1.5, 5.0, float(st.session_state.profile_data.get('walk_speed', 3.1)), 0.1)
with col_f2:
    p_run = st.select_slider("Running Pace (min/mi)", options=[f"{m}:{s:02d}" for m in range(5, 16) for s in [0, 30]], value=st.session_state.profile_data.get('run_pace', "9:00"))
    v_run = pace_to_mph(p_run)
with col_f3:
    v_vo2 = st.slider("VO2 Max", 15.0, 75.0, float(st.session_state.profile_data.get('vo2', 42.0)), 0.5)

st.divider()
col1_pa, col2_pa, col3_pa = st.columns(3)
SPORTS = ["None"] + sorted(["Walking", "Weight Lifting", "Bicycling", "Running", "Swimming", "Yoga", "Hiking", "Soccer"])
with col1_pa:
    s1 = st.selectbox("Activity 1", SPORTS, index=0)
    d1 = st.slider("Days/Wk 1", 0, 7, 5)
with col2_pa:
    s2 = st.selectbox("Activity 2", SPORTS, index=0)
    d2 = st.slider("Days/Wk 2", 0, 7, 0)
with col3_pa:
    s3 = st.selectbox("Activity 3", SPORTS, index=0)
    d3 = st.slider("Days/Wk 3", 0, 7, 0)

if st.button("🚀 CALCULATE BIOLOGICAL AGE", type="primary", use_container_width=True):
    model, scaler, feature_names = load_assets()
    if model:
        weight_kg, height_cm = weight_lb * 0.453592, (h_ft * 30.48) + (h_in * 2.54)
        bmi = weight_kg / ((height_cm/100)**2) if height_cm > 0 else 0
        
        in_d = {f: 0.0 for f in feature_names}
        in_d.update({'bpsys': sys_bp, 'bpdia': dia_bp, 'bmxwt': weight_kg, 'bmxht': height_cm, 'bmi': bmi, 'bmxpulse': pulse, 'waist': waist_in*2.54, 'pct_bft': pct_bft, 'crp': v_crp, 'trig': v_trig, 'ldl': v_ldl, 'hdl': v_hdl, 'gluc': v_gluc, 'alb': v_alb, 'iron': v_iron})
        
        df_in = pd.DataFrame([in_d])[feature_names]
        try: df_s = pd.DataFrame(scaler.transform(df_in), columns=feature_names)
        except: df_s = df_in
        
        pred = model.predict(df_s)[0]
        st.balloons()
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Biological Age", f"{pred:.1f} yrs", f"{pred-chrono_age:.1f} vs Chrono", delta_color="inverse")
        
        with c_res2:
            st.markdown("### 🎯 Longevity Insights")
            impacts = [
                (pred - model.predict(pd.DataFrame(scaler.transform(df_in.assign(waist=in_d['waist']-5.0)), columns=feature_names))[0], "Reduce waist by 2 inches"),
                (pred - model.predict(pd.DataFrame(scaler.transform(df_in.assign(bpsys=sys_bp-10)), columns=feature_names))[0], "Lower Systolic BP by 10 pts"),
                (1.2 if v_walk < 3.5 else 0.3, "Increase walking speed to 3.5+ mph (Major indicator)"),
                (0.8 if v_vo2 < 45 else 0.2, "Improve VO2 Max by 5 points")
            ]
            for imp, desc in sorted(impacts, key=lambda x: x[0], reverse=True)[:3]:
                if imp > 0.1: st.markdown(f'<div class="recommendation-card"><strong>-{imp:.1f} yr</strong>: {desc}</div>', unsafe_allow_html=True)
