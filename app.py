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

    /* FORCED BLUE SLIDERS - More thorough override */
    div[data-testid="stSlider"] [data-testid="stThumb"] {
        background-color: #3b82f6 !important;
        border: 2px solid #1d4ed8 !important;
    }
    div[data-testid="stSlider"] [data-val="true"] { 
        background-color: #3b82f6 !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #3b82f6 !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #3b82f6 !important;
    }

    /* High-contrast Text Inputs */
    input[type="number"], input[type="text"], div[data-baseweb="input"] {
        background-color: #1e40af !important; /* Deep Navy Blue */
        color: #ffffff !important; /* Pure White Text */
        font-weight: 600 !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 6px !important;
    }
    input::placeholder { color: #cbd5e1 !important; }

    /* Uniform Keyboard-style Buttons */
    div.stButton > button {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0) !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 5px 0 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 0 #94a3b8, 0 5px 10px rgba(0,0,0,0.05) !important;
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
    except: return 6.67

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
    's2': "None", 'd2': 0, 'i2': "Moderate",
    's3': "None", 'd3': 0, 'i3': "Moderate",
    'has_crp': False, 'has_trig': False, 'has_ldl': False, 'has_hdl': False,
    'has_gluc': False, 'has_alb': False, 'has_iron': False,
    'walk_speed': 3.1, 'run_pace': "9:00"
}
MARKER_DEFAULTS = {'trig': 80.0, 'ldl': 90.0, 'hdl': 60.0, 'gluc': 88.0, 'crp': 0.1, 'alb': 4.5, 'iron': 100.0}

if 'profile_data' not in st.session_state: st.session_state.profile_data = DEFAULT_STATE.copy()
if 'cur_initials' not in st.session_state: st.session_state.cur_initials = ""
if 'cur_birth_year' not in st.session_state: st.session_state.cur_birth_year = 1980

st.markdown("<h1>🧬 Biological Age Predictor</h1>", unsafe_allow_html=True)

with st.expander("👤 User Profile"):
    col_init, col_year = st.columns(2)
    i_in = col_init.text_input("Initials (3)", st.session_state.cur_initials, max_chars=3).upper()
    y_in = col_year.number_input("Birth Year", 1920, 2024, st.session_state.cur_birth_year)
    if st.button("Load Profile"):
        if len(i_in) == 3:
            st.session_state.cur_initials, st.session_state.cur_birth_year = i_in, y_in
            data = load_profile(i_in, y_in)
            if data: st.session_state.profile_data.update(data)
            st.rerun()

def multi_step_input(label, key, min_val, max_val, step_small=0.5, step_large=5.0, enabled=True):
    if key not in st.session_state: st.session_state[key] = float(st.session_state.profile_data.get(key, min_val))
    if not enabled:
        st.write(f"**{label}** (Auto)")
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
    w_lb = multi_step_input("Weight (lbs)", "weight", 80.0, 500.0, 1.0, 5.0)
    h_ft = st.selectbox("Feet", range(4, 9), index=int(st.session_state.profile_data.get('h_ft', 5)-4))
    h_in = st.selectbox("Inches", range(12), index=int(st.session_state.profile_data.get('h_in', 10)))
    waist = multi_step_input("Waist (in)", "waist_in", 20.0, 70.0, 0.5, 5.0)
    bft = multi_step_input("Body Fat %", "pct_bft", 3.0, 60.0, 0.5, 5.0)
    sys = multi_step_input("Systolic BP", "sys", 80.0, 200.0, 1.0, 5.0)
    dia = multi_step_input("Diastolic BP", "dia", 40.0, 120.0, 1.0, 5.0)
    pul = multi_step_input("Pulse", "pulse", 40.0, 150.0, 1.0, 5.0)
    st.divider()
    st.markdown("### 🩺 Labs")
    def lb(l, k, mk):
        h = st.checkbox(f"Have {l}?", st.session_state.profile_data.get(mk, False), key=mk)
        return multi_step_input(l, k, 0.0, 500.0, 1.0, 10.0, enabled=h), h
    v_crp, h_crp = lb("CRP", "crp", "has_crp")
    v_trig, h_trig = lb("Trig", "trig", "has_trig")
    v_ldl, h_ldl = lb("LDL", "ldl", "has_ldl")
    v_hdl, h_hdl = lb("HDL", "hdl", "has_hdl")
    v_gluc, h_gluc = lb("Glucose", "gluc", "has_gluc")
    v_alb, h_alb = lb("Albumin", "alb", "has_alb")
    v_iron, h_iron = lb("Iron", "iron", "has_iron")

st.subheader("🏆 Fitness & Mobility Indicators")
col_g1, col_g2, col_g3 = st.columns(3)
with col_g1: v_walk = st.slider("Walking Speed (mph)", 1.5, 5.0, float(st.session_state.profile_data.get('walk_speed', 3.1)), 0.1, key="walk_sl")
with col_g2:
    # Run pace: 5:00 - 15:45 min/mi, 15-sec increments
    run_pace_options = [f"{m}:{s:02d}" for m in range(5, 16) for s in [0, 15, 30, 45]]
    # Default to 9:00 if not in profile, clamp to valid options
    default_run = st.session_state.profile_data.get('run_pace', '9:00')
    if default_run not in run_pace_options:
        default_run = '9:00'
    p_run = st.select_slider("Running Pace (min/mi)", options=run_pace_options, value=default_run, key="run_sl")
    # Convert run pace → run speed mph, display it
    try:
        rm, rs = map(int, p_run.split(':'))
        run_speed_mph = 60.0 / (rm + rs/60.0)
        st.caption(f"≈ {run_speed_mph:.1f} mph")
    except: run_speed_mph = None
with col_g3: v_vo2 = st.slider("VO2 Max Value", 15.0, 75.0, float(st.session_state.profile_data.get('vo2', 42.0)), 0.5, key="vo2_sl")

st.markdown("### 🏃‍♂️ Sports & Activity Frequencies")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    s1 = st.selectbox("Primary Activity", ["Walking", "Running", "Cycling", "Swimming", "Weightlifting", "Tennis", "Basketball", "Soccer", "Other", "None"], index=0)
    d1 = st.slider("Days/Week (Primary)", 0, 7, 5)
    i1 = st.select_slider("Intensity (Primary)", options=["Low", "Moderate", "Vigorous"], value="Moderate")
with col_s2:
    s2 = st.selectbox("Secondary Activity", ["Walking", "Running", "Cycling", "Swimming", "Weightlifting", "Tennis", "Basketball", "Soccer", "Other", "None"], index=9)
    d2 = st.slider("Days/Week (Secondary)", 0, 7, 0)
    i2 = st.select_slider("Intensity (Secondary)", options=["Low", "Moderate", "Vigorous"], value="Moderate")
with col_s3:
    s3 = st.selectbox("Other Sport/Activity", ["Walking", "Running", "Cycling", "Swimming", "Weightlifting", "Tennis", "Basketball", "Soccer", "Other", "None"], index=9)
    d3 = st.slider("Days/Week (Other)", 0, 7, 0)
    i3 = st.select_slider("Intensity (Other)", options=["Low", "Moderate", "Vigorous"], value="Moderate")

if st.button("🚀 CALCULATE BIOLOGICAL AGE", type="primary", use_container_width=True):
    model, scaler, feature_names = load_assets()
    if model:
        in_d = {f: 0.0 for f in feature_names}
        w_kg, h_cm = w_lb * 0.453592, (h_ft * 30.48) + (h_in * 2.54)
        bmi_val = w_kg/((h_cm/100)**2)
        # Map UI inputs → model feature names (NHANES schema)
        # Vitals
        in_d.update({
            'sys_bp': sys, 'dia_bp': dia,
            'wtkg': w_kg, 'ht_cm': h_cm, 'bmi': bmi_val,
            'pulse': pul,
            'waist': waist*2.54, 'pct_bft': bft,
            # Labs
            'crp': v_crp, 'trig': v_trig, 'ldl': v_ldl, 'hdl': v_hdl,
            'gluc': v_gluc, 'alb': v_alb, 'iron': v_iron,
        })
        # Fitness / mobility — map UI → NHANES model features
        # walk_t: NHANES 20-ft walk test time (sec). Approx from walking speed.
        # 20 ft ≈ 0.003788 mi. time(sec) = distance/speed * 3600
        # If user entered a run pace, estimate walk speed from it too.
        # Typical walk/run speed ratio ≈ 0.38-0.45 for fit adults.
        # Use the faster of: direct walk_speed, or run_speed * 0.40
        try:
            rm, rs = map(int, p_run.split(':'))
            run_speed_mph = 60.0 / (rm + rs/60.0)
        except:
            run_speed_mph = None
        # Estimate walk speed from run pace (run * 0.40), use max of walk_slider vs run-derived
        # This way a fast runner isn't penalized by a conservative walk_speed slider
        walk_mph_est = run_speed_mph * 0.40 if run_speed_mph else v_walk
        effective_walk_mph = max(v_walk, walk_mph_est)
        # Clamp to slider range
        effective_walk_mph = min(max(effective_walk_mph, 1.5), 5.0)
        walk_t_sec = (0.003788 / max(effective_walk_mph, 0.5)) * 3600
        in_d['walk_t'] = walk_t_sec
        in_d['vo2_max'] = v_vo2
        # Top model features not in UI — use population-typical defaults
        # so predictions aren't skewed by 0-defaults
        # cogn: cognitive score, tel: telomere length proxy,
        # blnc_3: 3-sec balance test, sob_stairs: SOB on stairs, swelling: leg swelling
        # Defaults set to healthy/normal population means
        in_d['cogn'] = in_d.get('cogn', 0.0) or 0.0  # z-score, 0 = population mean
        in_d['tel'] = 1.0       # telomere T/S ratio, ~1.0 is normal
        in_d['blnc_3'] = 1.0    # 1 = pass 3-sec balance
        in_d['sob_stairs'] = 0.0  # 0 = no SOB
        in_d['swelling'] = 0.0    # 0 = no leg swelling
        # Activity mapping: UI sports → model ACT_* features + activity counters
        # Map activity names to model feature names
        act_map = {
            'Weightlifting': 'ACT_WEIGHT LIFTING',
            'Walking': 'ACT_WALKING',
            'Running': 'ACT_RUNNING',
            'Cycling': 'ACT_BICYCLING',
            'Swimming': 'ACT_SWIMMING',
            'Tennis': 'ACT_TENNIS',
            'Basketball': 'ACT_BASKETBALL',
            'Soccer': 'ACT_SOCCER',
        }
        # Set activity flags (days/week as weight)
        for sport, days, inten in [(s1, d1, i1), (s2, d2, i2), (s3, d3, i3)]:
            if sport != 'None' and sport in act_map:
                feat = act_map[sport]
                if feat in in_d:
                    in_d[feat] = max(in_d[feat], float(days))
        # Activity frequency counters (days per 30 days)
        for sport, days, inten in [(s1, d1, i1), (s2, d2, i2), (s3, d3, i3)]:
            if sport != 'None' and days > 0:
                days_30 = days * 30 / 7
                if inten == 'Vigorous':
                    in_d['vig_act_30'] = in_d.get('vig_act_30', 0) + days_30
                elif inten == 'Moderate':
                    in_d['mod_act_30'] = in_d.get('mod_act_30', 0) + days_30
                if sport == 'Weightlifting':
                    in_d['strength_30'] = in_d.get('strength_30', 0) + days_30
        
        def get_pred(data_dict):
            df = pd.DataFrame([data_dict])[feature_names]
            return model.predict(pd.DataFrame(scaler.transform(df), columns=feature_names))[0]

        pred = get_pred(in_d)
        c1, c2 = st.columns(2)
        c1.metric("Biological Age", f"{pred:.1f} yrs", f"{pred-chrono_age:.1f} vs Chrono", delta_color="inverse")
        
        with c2:
            st.markdown("### 🎯 Longevity Insights")
            # Test real feature impacts (using correct feature names)
            # walk_t is the #1 predictor (50.8% importance) — derived from v_walk
            walk_t_faster = (0.003788 / max(min(v_walk + 0.5, 5.0), 0.5)) * 3600
            imp = [
                (pred - get_pred({**in_d, 'waist': (waist-2)*2.54}), "Reduce waist by 2 inches"),
                (pred - get_pred({**in_d, 'sys_bp': sys-10}), "Lower Systolic BP by 10 pts"),
                (pred - get_pred({**in_d, 'walk_t': walk_t_faster}), "Improve walking speed by 0.5 MPH"),
                (pred - get_pred({**in_d, 'ldl': max(v_ldl-20, 50)}), "Lower LDL by 20 mg/dL"),
                (0.9 if v_vo2 < 45 else 0.2, "Improve VO2 Max by 5 pts")
            ]
            for val, desc in sorted(imp, key=lambda x: x[0], reverse=True)[:3]:
                if val > 0.05: st.markdown(f'<div class="recommendation-card"><strong>-{val:.1f} yr</strong>: {desc}</div>', unsafe_allow_html=True)
