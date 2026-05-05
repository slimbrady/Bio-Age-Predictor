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

    /* Realistic 3D Buttons */
    div.stButton > button {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb) !important;
        color: #0d47a1 !important;
        border: 1px solid #90caf9 !important;
        border-radius: 6px !important;
        padding: 8px 0 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 0 #64b5f6, 
                    0 5px 10px rgba(0,0,0,0.1) !important;
        transition: all 0.1s ease !important;
        width: 100% !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(145deg, #bbdefb, #90caf9) !important;
        box-shadow: 0 2px 0 #42a5f5, 
                    0 3px 6px rgba(0,0,0,0.1) !important;
        transform: translateY(2px) !important;
    }
    
    div.stButton > button:active {
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
        transform: translateY(4px) !important;
    }

    /* Shaded state for disabled inputs */
    .shaded-input {
        opacity: 0.5;
        filter: grayscale(100%);
    }

    [data-testid="stExpander"], [data-testid="column"], [data-testid="stVerticalBlock"] > div:has(div.stExpander) {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid #f1f5f9;
    }

    div[data-baseweb="input"] {
        height: 42px !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- USER PROFILE PERSISTENCE ---
def get_user_file(initials, birth_year):
    if not initials or len(str(initials)) < 3: return None
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, f"profile_{str(initials).lower()}_{birth_year}.json")

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

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_dir, "xgb_bioage_model.joblib"))
        scaler = joblib.load(os.path.join(base_dir, "scaler_x.pkl"))
        features = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
        return model, scaler, features
    except: return None, None, None

# --- INITIALIZE DEFAULTS ---
DEFAULT_STATE = {
    'age': 45.0, 'weight': 145.0, 'h_ft': 5, 'h_in': 10,
    'waist_in': 34.0, 'pct_bft': 18.0, 'sys': 115.0, 'dia': 75.0, 'pulse': 60.0,
    'vo2': 52.0, 'crp': 0.1, 'trig': 80.0, 'ldl': 90.0, 'hdl': 60.0,
    'gluc': 88.0, 'alb': 4.5, 'iron': 100.0, 's1': "Walking", 'd1': 5, 'i1': "Moderate",
    's2': "Running", 'd2': 4, 'i2': "Vigorous", 's3': "None", 'd3': 0, 'i3': "Moderate",
    'run_pace': "8:00",
    'has_crp': True, 'has_trig': True, 'has_ldl': True, 'has_hdl': True,
    'has_gluc': True, 'has_alb': True, 'has_iron': True
}

if 'profile_data' not in st.session_state: st.session_state.profile_data = DEFAULT_STATE.copy()
if 'user_loaded' not in st.session_state: st.session_state.user_loaded = False
if 'cur_initials' not in st.session_state: st.session_state.cur_initials = ""
if 'cur_birth_year' not in st.session_state: st.session_state.cur_birth_year = 1980

# --- HEADER ---
st.markdown("<h1>🧬 Biological Age Predictor</h1>", unsafe_allow_html=True)

# --- LOGIN SECTION ---
with st.expander("👤 User Login / Profile Management", expanded=not st.session_state.user_loaded):
    col_init, col_year = st.columns(2)
    initials_input = col_init.text_input("Initials (3 chars)", value=st.session_state.cur_initials, max_chars=3).upper()
    birth_year_input = col_year.number_input("Year of Birth", 1920, 2024, value=st.session_state.cur_birth_year)
    
    if st.button("Load Saved Profile"):
        if len(initials_input) < 3:
            st.error("Please enter exactly 3 initials.")
        else:
            data = load_profile(initials_input, birth_year_input)
            st.session_state.cur_initials = initials_input
            st.session_state.cur_birth_year = birth_year_input
            if data:
                st.session_state.profile_data.update(data)
                for k, v in data.items():
                    if k in st.session_state: st.session_state[k] = v
                st.session_state.user_loaded = True
                st.rerun()
            else:
                st.session_state.profile_data = DEFAULT_STATE.copy()
                st.session_state.user_loaded = True
                st.rerun()

# --- INPUT HELPER WITH ROBUST STEP LOGIC ---
def multi_step_input(label, key, min_val, max_val, step_small=0.5, step_large=5.0, enabled=True):
    if key not in st.session_state:
        st.session_state[key] = float(st.session_state.profile_data.get(key, DEFAULT_STATE.get(key, min_val)))

    if not enabled:
        st.markdown(f"<div class='shaded-input'><strong>{label} (Omitted)</strong></div>", unsafe_allow_html=True)
        st.caption("Value will use clinical average in calculation")
        return None

    st.write(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    l1, l2, l3, l4 = f"-{int(step_large)}", f"-{step_small}", f"+{step_small}", f"+{int(step_large)}"
    if not step_large.is_integer(): l1, l4 = f"-{step_large}", f"+{step_large}"

    if c1.button(l1, key=f"btn1_{key}"): 
        st.session_state[key] = float(max(min_val, st.session_state[key] - step_large))
        st.rerun()
    if c2.button(l2, key=f"btn2_{key}"): 
        st.session_state[key] = float(max(min_val, st.session_state[key] - step_small))
        st.rerun()
    if c3.button(l3, key=f"btn3_{key}"): 
        st.session_state[key] = float(min(max_val, st.session_state[key] + step_small))
        st.rerun()
    if c4.button(l4, key=f"btn4_{key}"): 
        st.session_state[key] = float(min(max_val, st.session_state[key] + step_large))
        st.rerun()

    # User requested + - in field to go up/down by 1
    val = st.number_input(label, min_val, max_val, step=1.0, key=key, label_visibility="collapsed")
    return val

# --- SIDEBAR inputs ---
with st.sidebar:
    st.markdown("### 📋 Primary Vitals")
    chrono_age = st.number_input("Chronological Age", 18, 100, int(st.session_state.profile_data.get('age', 45)))
    weight_lb = multi_step_input("Weight (lbs)", "weight", 80.0, 500.0, 1.0, 5.0)
    
    st.write("**Height**")
    h_ft = st.selectbox("Feet", range(4, 9), index=(int(st.session_state.profile_data.get('h_ft', 5)) - 4))
    h_in = st.selectbox("Inches", range(12), index=int(st.session_state.profile_data.get('h_in', 10)))
    
    waist_in = multi_step_input("Waist (in)", "waist_in", 20.0, 70.0, 0.5, 5.0)
    pct_bft = multi_step_input("Body Fat %", "pct_bft", 3.0, 60.0, 0.5, 5.0)
    sys_bp = multi_step_input("Systolic BP", "sys", 80.0, 200.0, 1.0, 5.0)
    dia_bp = multi_step_input("Diastolic BP", "dia", 40.0, 120.0, 1.0, 5.0)
    pulse = multi_step_input("Pulse", "pulse", 40.0, 150.0, 1.0, 5.0)

    st.divider()
    st.markdown("### 🩺 Lab Markers")
    def lab_toggle_input(label, key, m_key, min_v, max_v, s_s, s_l):
        has_val = st.checkbox(f"Know my {label}?", value=st.session_state.profile_data.get(m_key, True), key=m_key)
        val = multi_step_input(label, key, min_v, max_v, s_s, s_l, enabled=has_val)
        return val, has_val

    v_crp, h_crp = lab_toggle_input("CRP", "crp", "has_crp", 0.0, 15.0, 0.1, 1.0)
    v_trig, h_trig = lab_toggle_input("Triglycerides", "trig", "has_trig", 0.0, 500.0, 1.0, 10.0)
    v_ldl, h_ldl = lab_toggle_input("LDL", "ldl", "has_ldl", 0.0, 350.0, 1.0, 10.0)
    v_hdl, h_hdl = lab_toggle_input("HDL", "hdl", "has_hdl", 0.0, 150.0, 1.0, 10.0)
    v_gluc, h_gluc = lab_toggle_input("Glucose", "gluc", "has_gluc", 0.0, 300.0, 1.0, 10.0)
    v_alb, h_alb = lab_toggle_input("Albumin", "alb", "has_alb", 0.0, 6.0, 0.1, 0.5)
    v_iron, h_iron = lab_toggle_input("Iron", "iron", "has_iron", 0.0, 300.0, 1.0, 10.0)

# --- PHYSICAL ACTIVITY ---
st.subheader("🏆 Physical Activity & Fitness")
col1_pa, col2_pa, col3_pa = st.columns(3)
DATASET_SPORTS = sorted(["BASKETBALL", "GARDENING", "YARD WORK", "WALKING", "WEIGHT LIFTING", "BICYCLING", "RUNNING", "AEROBICS", "PUSH-UPS", "FOOTBALL", "ROLLERBLADING", "BOWLING", "TENNIS", "DANCE", "SOCCER", "JOGGING", "STAIR CLIMBING", "SIT-UPS", "ROPE JUMPING", "HIKING", "SWIMMING", "BOXING", "MARTIAL ARTS", "GOLF", "VOLLEYBALL", "FISHING", "BASEBALL", "STRETCHING", "FRISBEE", "YOGA", "CHEERLEADING", "RACQUETBALL", "WRESTLING", "SOFTBALL", "HOCKEY", "TREADMILL", "SKIING", "SKATING", "SURFING", "SKATEBOARDING"])
SPORTS_OPTIONS = ["None"] + [s.title() for s in DATASET_SPORTS]

with col1_pa:
    s1 = st.selectbox("Activity 1", SPORTS_OPTIONS, key="s1")
    d1 = st.slider("Days/Week (1)", 0, 7, int(st.session_state.get('d1', 5)))
    i1 = st.select_slider("Intensity (1)", ["Light", "Moderate", "Vigorous"], value="Moderate")
with col2_pa:
    s2 = st.selectbox("Activity 2", SPORTS_OPTIONS, key="s2")
    d2 = st.slider("Days/Week (2)", 0, 7, int(st.session_state.get('d2', 0)))
    i2 = st.select_slider("Intensity (2)", ["Light", "Moderate", "Vigorous"], value="Moderate")
with col3_pa:
    s3 = st.selectbox("Activity 3", SPORTS_OPTIONS, key="s3")
    d3 = st.slider("Days/Week (3)", 0, 7, int(st.session_state.get('d3', 0)))
    i3 = st.select_slider("Intensity (3)", ["Light", "Moderate", "Vigorous"], value="Moderate")

# --- CALCULATION ---
st.divider()
if st.button("🚀 CALCULATE BIOLOGICAL AGE", use_container_width=True):
    model, scaler, feature_names = load_assets()
    if model:
        weight_kg = weight_lb * 0.453592
        height_cm = (h_ft * 30.48) + (h_in * 2.54)
        bmi = weight_kg / ((height_cm/100)**2)
        MARKER_DEFAULTS = {'trig': 80, 'ldl': 90, 'hdl': 60, 'gluc': 88, 'crp': 0.1, 'alb': 4.5, 'iron': 100}
        
        input_dict = {f: 0.0 for f in feature_names}
        input_dict['bpsys'], input_dict['bpdia'] = sys_bp, dia_bp
        input_dict['bmxwt'], input_dict['bmxht'] = weight_kg, height_cm
        input_dict['bmi'], input_dict['bmxpulse'] = bmi, pulse
        input_dict['waist'], input_dict['pct_bft'] = waist_in * 2.54, pct_bft
        
        input_dict['crp'] = v_crp if h_crp else MARKER_DEFAULTS['crp']
        input_dict['trig'] = v_trig if h_trig else MARKER_DEFAULTS['trig']
        input_dict['ldl'] = v_ldl if h_ldl else MARKER_DEFAULTS['ldl']
        input_dict['hdl'] = v_hdl if h_hdl else MARKER_DEFAULTS['hdl']
        input_dict['gluc'] = v_gluc if h_gluc else MARKER_DEFAULTS['gluc']
        input_dict['alb'] = v_alb if h_alb else MARKER_DEFAULTS['alb']
        input_dict['iron'] = v_iron if h_iron else MARKER_DEFAULTS['iron']
        
        df_in = pd.DataFrame([input_dict])[feature_names]
        try: df_in = pd.DataFrame(scaler.transform(df_in), columns=feature_names)
        except: pass
        bio_prediction = model.predict(df_in)[0]
        st.metric("Biological Age Estimate", f"{bio_prediction:.1f} yrs", delta=f"{bio_prediction - chrono_age:.1f} yrs", delta_color="inverse")
    else: st.error("Model assets missing.")

if st.button("💾 SAVE THIS PROFILE", use_container_width=True):
    if len(st.session_state.cur_initials) == 3:
        final_save = {k: st.session_state[k] for k in DEFAULT_STATE.keys() if k in st.session_state}
        save_profile(final_save, st.session_state.cur_initials, st.session_state.cur_birth_year)
        st.success(f"Profile for {st.session_state.cur_initials} saved!")
