import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bio-Age Calculator", page_icon="🧬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stNumberInput div[data-baseweb="input"] { height: 45px; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- USER PROFILE PERSISTENCE ---
def get_user_file(initials, birth_year):
    if not initials or len(str(initials)) < 3:
        return None
    base_dir = os.path.dirname(__file__)
    filename = f"profile_{str(initials).lower()}_{birth_year}.json"
    return os.path.join(base_dir, filename)

def save_profile(data, initials, birth_year):
    path = get_user_file(initials, birth_year)
    if path:
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except:
            pass

def load_profile(initials, birth_year):
    path = get_user_file(initials, birth_year)
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
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
    except:
        return None, None, None

# --- UI DESIGN ---
st.title("🧬 Biological Age & Longevity Predictor")

# --- INITIALIZE DEFAULTS ---
DEFAULT_STATE = {
    'age': 45.0, 'weight': 145.0, 'h_ft': 5, 'h_in': 10,
    'waist_in': 34.0, 'pct_bft': 18.0, 'sys': 115.0, 'dia': 75.0, 'pulse': 60.0,
    'vo2': 52.0, 'crp': 0.1, 'trig': 80.0, 'ldl': 90.0, 'hdl': 60.0,
    'gluc': 88.0, 'alb': 4.5, 'iron': 100.0, 's1': "Walking", 'd1': 5,
    's2': "Running", 'd2': 4, 's3': "None", 'd3': 0, 'run_pace': "8:00"
}

# Ensure EVERYTHING in profile_data exists to prevent KeyError
if 'profile_data' not in st.session_state:
    st.session_state.profile_data = DEFAULT_STATE.copy()

if 'user_loaded' not in st.session_state: st.session_state.user_loaded = False
if 'cur_initials' not in st.session_state: st.session_state.cur_initials = ""
if 'cur_birth_year' not in st.session_state: st.session_state.cur_birth_year = 1980

# --- LOGIN SECTION ---
with st.expander("👤 User Login / Load Profile", expanded=not st.session_state.user_loaded):
    col_init, col_year = st.columns(2)
    initials_input = col_init.text_input("Initials (3 characters)", value=st.session_state.cur_initials, max_chars=3, placeholder="ABC").upper()
    birth_year_input = col_year.number_input("Year of Birth", 1920, 2024, value=st.session_state.cur_birth_year)
    
    if st.button("Load My Data"):
        if len(initials_input) < 3:
            st.error("⚠️ Please enter exactly 3 initials.")
        else:
            data = load_profile(initials_input, birth_year_input)
            st.session_state.cur_initials = initials_input
            st.session_state.cur_birth_year = birth_year_input
            
            # Clear input sync-keys to force re-initialization
            for k in list(st.session_state.keys()):
                if k.startswith("input_val_"): del st.session_state[k]
                if k.startswith("wid_"): del st.session_state[k]
                
            if data:
                # Merge loaded data with defaults to ensure no missing keys
                new_state = DEFAULT_STATE.copy()
                new_state.update(data)
                st.session_state.profile_data = new_state
                st.session_state.user_loaded = True
                st.success("Welcome back! Data loaded.")
                st.rerun()
            else:
                st.session_state.profile_data = DEFAULT_STATE.copy()
                st.warning(f"No profile found for {initials_input}. Starting fresh.")
                st.session_state.user_loaded = True
                st.rerun()

# --- INPUT HELPER ---
def multi_step_input(label, key, min_val, max_val, default_val, is_float=False):
    sync_key = f"input_val_{key}"
    if sync_key not in st.session_state:
        # PULL FROM PROFILE DATA
        val = st.session_state.profile_data.get(key, default_val)
        st.session_state[sync_key] = float(val)
    
    st.write(f"**{label}**")
    
    # Large number input
    curr_val = st.number_input(label, float(min_val), float(max_val), float(st.session_state[sync_key]), 
                              label_visibility="collapsed", key=f"wid_{key}", step=0.1 if is_float else 1.0)
    st.session_state[sync_key] = curr_val

    # Buttons
    c1, c2, c3, c4 = st.columns(4)
    step_l = 5.0
    step_s = 0.5 if is_float else 1.0
    
    if c1.button(f"-{step_l}", key=f"btn_m5_{key}"):
        st.session_state[sync_key] = max(float(min_val), round(st.session_state[sync_key] - step_l, 2))
        st.rerun()
    if c2.button(f"-{step_s}", key=f"btn_m1_{key}"):
        st.session_state[sync_key] = max(float(min_val), round(st.session_state[sync_key] - step_s, 2))
        st.rerun()
    if c3.button(f"+{step_s}", key=f"btn_p1_{key}"):
        st.session_state[sync_key] = min(float(max_val), round(st.session_state[sync_key] + step_s, 2))
        st.rerun()
    if c4.button(f"+{step_l}", key=f"btn_p5_{key}"):
        st.session_state[sync_key] = min(float(max_val), round(st.session_state[sync_key] + step_l, 2))
        st.rerun()
        
    return st.session_state[sync_key]

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Personal Profile")
    
    with st.expander("Vitals", expanded=True):
        chrono_age = st.number_input("Chronological Age", 18, 100, int(st.session_state.profile_data.get('age', 45)))
        weight_lb = multi_step_input("Weight (lbs)", "weight", 80, 500, 145)
        weight_kg = weight_lb * 0.453592
        
        st.write("**Height**")
        h_ft = st.selectbox("Feet", range(4, 9), index=(int(st.session_state.profile_data.get('h_ft', 5)) - 4))
        h_in = st.selectbox("Inches", range(12), index=int(st.session_state.profile_data.get('h_in', 10)))
        total_inches = (h_ft * 12) + h_in
        height_cm = total_inches * 2.54
        
        bmi = weight_kg / ((height_cm/100)**2) if height_cm > 0 else 0
        st.caption(f"BMI: {bmi:.1f}")
        
        waist_in = multi_step_input("Waist (in)", "waist_in", 20.0, 70.0, 34.0, is_float=True)
        pct_bft = multi_step_input("Body Fat %", "pct_bft", 3.0, 60.0, 18.0, is_float=True)
        sys_bp = multi_step_input("Systolic BP", "sys", 80, 200, 115)
        dia_bp = multi_step_input("Diastolic BP", "dia", 40, 120, 75)
        pulse = multi_step_input("Pulse", "pulse", 40, 150, 60)

    with st.expander("Performance", expanded=True):
        vo2_max = st.slider("VO2 Max", 15.0, 85.0, float(st.session_state.profile_data.get('vo2', 52.0)))
        perf_type = st.radio("Input Style:", ["Walking Speed (mph)", "Running Pace (min/mile)"])
        if perf_type == "Walking Speed (mph)":
            walk_mph = st.slider("Walking Speed", 1.0, 5.0, 3.5)
            walk_t_score = 6.0 / (walk_mph * 0.44704)
        else:
            pace_options = [f"{m}:{s:02d}" for m in range(4, 21) for s in range(0, 60, 5)]
            saved_pace = str(st.session_state.profile_data.get('run_pace', "8:00"))
            try: p_idx = pace_options.index(saved_pace)
            except: p_idx = pace_options.index("8:00")
            sel_pace = st.select_slider("Running Pace", options=pace_options, value=saved_pace)
            parts = sel_pace.split(':')
            m, s = int(parts[0]), int(parts[1])
            run_min_mile = m + s/60.0
            walk_t_score = 6.0 / ((60.0 / run_min_mile) * 0.44704)

    with st.expander("🩺 Lab Markers"):
        lab_crp = multi_step_input("CRP", "crp", 0.0, 15.0, 0.0, is_float=True)
        lab_trig = multi_step_input("Triglycerides", "trig", 0, 500, 0)
        lab_ldl = multi_step_input("LDL Chol", "ldl", 0, 350, 0)
        lab_hdl = multi_step_input("HDL Chol", "hdl", 0, 150, 0)
        lab_gluc = multi_step_input("Glucose", "gluc", 0, 300, 0)
        lab_alb = multi_step_input("Albumin", "alb", 0.0, 6.0, 0.0, is_float=True)
        lab_iron = multi_step_input("Iron", "iron", 0, 300, 0)

# --- MAIN AREA ---
st.subheader("🏆 Physical Activity")
col1_pa, col2_pa, col3_pa = st.columns(3)
p_s1 = st.session_state.profile_data.get('s1', "Walking")
p_s2 = st.session_state.profile_data.get('s2', "Running")
p_s3 = st.session_state.profile_data.get('s3', "None")

DATASET_SPORTS = sorted(["BASKETBALL", "GARDENING", "YARD WORK", "WALKING", "WEIGHT LIFTING", "BICYCLING", "RUNNING", "AEROBICS", "PUSH-UPS", "FOOTBALL", "ROLLERBLADING", "BOWLING", "TENNIS", "DANCE", "SOCCER", "JOGGING", "STAIR CLIMBING", "SIT-UPS", "ROPE JUMPING", "HIKING", "SWIMMING", "BOXING", "MARTIAL ARTS", "GOLF", "VOLLEYBALL", "FISHING", "BASEBALL", "STRETCHING", "FRISBEE", "YOGA", "CHEERLEADING", "RACQUETBALL", "WRESTLING", "SOFTBALL", "HOCKEY", "TREADMILL", "SKIING", "SKATING", "SURFING", "SKATEBOARDING"])
SPORTS_OPTIONS = ["None"] + [s.title() for s in DATASET_SPORTS]

with col1_pa:
    s1 = st.selectbox("Activity 1", SPORTS_OPTIONS, index=SPORTS_OPTIONS.index(p_s1) if p_s1 in SPORTS_OPTIONS else 0)
    d1 = st.slider("Days/Week (A1)", 0, 7, int(st.session_state.profile_data.get('d1', 5)))
with col2_pa:
    s2 = st.selectbox("Activity 2", SPORTS_OPTIONS, index=SPORTS_OPTIONS.index(p_s2) if p_s2 in SPORTS_OPTIONS else 0)
    d2 = st.slider("Days/Week (A2)", 0, 7, int(st.session_state.profile_data.get('d2', 4)))
with col3_pa:
    s3 = st.selectbox("Activity 3", SPORTS_OPTIONS, index=SPORTS_OPTIONS.index(p_s3) if p_s3 in SPORTS_OPTIONS else 0)
    d3 = st.slider("Days/Week (A3)", 0, 7, int(st.session_state.profile_data.get('d3', 0)))

paq_val = min((d1*5 + d2*5 + d3*5) * 1.5, 30.0)

st.divider()
c_calc, c_save = st.columns(2)

if c_calc.button("🚀 Calculate BIO-AGE", use_container_width=True):
    model, scaler, feature_names = load_assets()
    if model:
        input_dict = {f: 0.0 for f in feature_names}
        direct_map = {
            'bpsys': sys_bp, 'bpdia': dia_bp, 'bmxwt': weight_kg, 'bmxht': height_cm, 
            'bmxpulse': pulse, 'walk_t': walk_t_score, 'final_vo2': vo2_max, 
            'paq': paq_val, 'bmi': bmi, 'waist': waist_in * 2.54, 'pct_bft': pct_bft,
            'crp': lab_crp, 'trig': lab_trig, 'ldl': lab_ldl, 'hdl': lab_hdl, 
            'gluc': lab_gluc, 'alb': lab_alb, 'iron': lab_iron
        }
        defaults = {'trig': 80, 'ldl': 90, 'hdl': 60, 'gluc': 88, 'crp': 0.1, 'alb': 4.5, 'iron': 100}

        for k, v in direct_map.items():
            matches = [f for f in feature_names if k.lower() == f.lower().split('_')[0] or k.lower() == f.lower()]
            for m in matches:
                input_dict[m] = v if (v > 0 or k not in defaults) else defaults[k]
        
        for act, d in [(s1, d1), (s2, d2), (s3, d3)]:
            if act != "None":
                fn = f"ACT_{act.upper()}"; m = [f for f in feature_names if fn in f]
                if m: input_dict[m[0]] = 1.0

        df_in = pd.DataFrame([input_dict])[feature_names]
        try: df_in = pd.DataFrame(scaler.transform(df_in), columns=feature_names)
        except: pass
        
        bio_prediction = model.predict(df_in)[0]
        st.metric("Model Predicted Age", f"{bio_prediction:.1f} yrs", delta=f"{bio_prediction - chrono_age:.1f} yrs", delta_color="inverse")
    else: st.error("Model assets not found.")

if c_save.button("💾 Save Profile", use_container_width=True):
    if len(st.session_state.cur_initials) < 3:
        st.error("⚠️ Please login with 3 initials first.")
    else:
        current_save = {
            'age': chrono_age, 'weight': weight_lb, 'h_ft': h_ft, 'h_in': h_in, 
            'sys': sys_bp, 'dia': dia_bp, 'pulse': pulse, 'vo2': vo2_max, 
            'waist_in': waist_in, 'pct_bft': pct_bft, 'crp': lab_crp,
            'trig': lab_trig, 'ldl': lab_ldl, 'hdl': lab_hdl, 'gluc': lab_gluc,
            'alb': lab_alb, 'iron': lab_iron, 's1': s1, 'd1': d1, 's2': s2, 'd2': d2,
            's3': s3, 'd3': d3, 'run_pace': (sel_pace if perf_type != "Walking Speed (mph)" else "8:00")
        }
        save_profile(current_save, st.session_state.cur_initials, st.session_state.cur_birth_year)
        st.success("Profile saved.")
