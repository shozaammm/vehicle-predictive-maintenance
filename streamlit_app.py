import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from datetime import datetime
from speedx_assets import CAR_B64

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vectis Telemetry // Predictive Maintenance Cockpit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# HIGH-END AUTOMOTIVE COCKPIT STYLING (Zero Emojis, Double-Bezel, Pure Obsidian)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-cockpit: #090B0E;
    --card-surface: #11141B;
    --card-surface-subtle: #171B24;
    --card-border: rgba(255, 255, 255, 0.08);
    --speedx-green: #22C55E;
    --speedx-green-glow: rgba(34, 197, 94, 0.35);
    --speedx-amber: #F59E0B;
    --speedx-red: #EF4444;
    --text-white: #FFFFFF;
    --text-muted: #8E9BAE;
    --text-dim: #64748B;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-cockpit) !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(34, 197, 94, 0.04) 0%, transparent 60%),
        radial-gradient(circle at 90% 90%, rgba(245, 158, 11, 0.02) 0%, transparent 50%) !important;
    color: var(--text-white) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: #0D1016 !important;
    border-right: 1px solid var(--card-border) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--card-border) !important;
    margin: 14px 0 !important;
}

/* Streamlit Inputs */
.stSlider [data-baseweb="slider"] {
    color: var(--speedx-green) !important;
}

div[data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--speedx-green) !important;
    font-weight: 700 !important;
}

div[data-testid="stTickBar"] {
    background: rgba(255, 255, 255, 0.08) !important;
}

/* Card Outer Shell (Double-Bezel Architecture) */
.speedx-card {
    background: var(--card-surface);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
}

.speedx-card-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.speedx-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Progress Bars */
.speedx-bar-container {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
}

.speedx-bar-fill {
    background: linear-gradient(90deg, #16A34A 0%, #22C55E 100%);
    box-shadow: 0 0 10px var(--speedx-green-glow);
    height: 100%;
    border-radius: 4px;
}

/* 5-Segment LED Bars */
.speedx-segments {
    display: flex;
    gap: 6px;
    margin-top: 8px;
}

.speedx-segment-block {
    flex: 1;
    height: 16px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 2px;
}

.speedx-segment-block.lit {
    background: var(--speedx-green);
    box-shadow: 0 0 8px var(--speedx-green-glow);
}

/* Engineering Advisory Box */
.speedx-quote-box {
    background: #151922;
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin: 10px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
    color: #E2E8F0;
    border-left: 3px solid var(--speedx-green);
}

/* Table Architecture */
.matrix-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

.matrix-table th {
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-muted);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
}

.matrix-table td {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.matrix-table tr:hover td {
    background: rgba(255, 255, 255, 0.02);
}

/* Streamlit Tabs Customization */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #11141B !important;
    padding: 6px !important;
    border-radius: 12px !important;
    gap: 8px !important;
    border: 1px solid var(--card-border) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8E9BAE !important;
    border: none !important;
    padding: 8px 18px !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background: #22C55E !important;
    color: #090B0E !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PREDICTIVE MAINTENANCE ENGINE (Mathematical Modeling)
# -----------------------------------------------------------------------------
class VehiclePredictiveMaintenanceModel:
    def __init__(self):
        pass

    def predict_maintenance_needs(self, data):
        rpm = data['Engine_RPM']
        temp = data['Coolant_Temperature']
        load = data['Engine_Load']
        speed = data['Vehicle_Speed']
        throttle = data['Throttle_Position']
        fuel = data['Fuel_Pressure']
        air = data['Air_Temperature']
        mileage = data['Mileage']

        risk_factors = []
        subsystem_issues = []

        # 1. Thermal Analysis
        if temp > 115:
            risk_factors.append(0.35)
            subsystem_issues.append("Cooling Loop Critical Overtemperature")
        elif temp > 102:
            risk_factors.append(0.18)
            subsystem_issues.append("Cooling Loop Elevated Thermal Strain")
        elif temp < 75:
            risk_factors.append(0.08)
            subsystem_issues.append("Coolant Sub-Optimal Temperature")

        # 2. RPM & Crankshaft Dynamics
        if rpm > 7000:
            risk_factors.append(0.25)
            subsystem_issues.append("High Crankshaft RPM Stress")
        elif rpm > 6000:
            risk_factors.append(0.12)

        # 3. Combustion / Manifold Load
        if load > 85:
            risk_factors.append(0.20)
            subsystem_issues.append("Manifold Mechanical Overload")
        elif load > 75:
            risk_factors.append(0.10)

        # 4. Common Rail Fuel Pressure
        if fuel < 34:
            risk_factors.append(0.30)
            subsystem_issues.append("Common Rail Injection Starvation (<34 PSI)")
        elif fuel > 65:
            risk_factors.append(0.20)
            subsystem_issues.append("Fuel Rail Overpressure Surge (>65 PSI)")

        # 5. Mileage Wear Factor
        mileage_factor = min(0.25, (mileage / 250000.0) * 0.25)
        risk_factors.append(mileage_factor)

        # Combined Risk Score (0.0 to 1.0)
        base_risk = sum(risk_factors)
        risk_score = min(0.98, max(0.04, base_risk))

        # Failure Probability
        failure_prob = 1.0 / (1.0 + math.exp(-6.0 * (risk_score - 0.45)))

        # Multi-variable Anomaly Score
        rpm_norm = (rpm - 1500) / 4500.0
        temp_norm = (temp - 90) / 25.0
        load_norm = (load - 50) / 30.0
        fuel_norm = abs(fuel - 45) / 15.0
        anomaly_score = min(1.0, math.sqrt(rpm_norm**2 + temp_norm**2 + load_norm**2 + fuel_norm**2) / 3.0)

        # Precise Engineering Recommendations
        if risk_score > 0.70:
            rec = "CRITICAL ADVISORY: Significant powertrain anomaly detected. High probability of thermal breakdown or fuel rail starvation. Halt full-throttle operation and dispatch workshop diagnostics."
            risk_level = "CRITICAL"
        elif risk_score > 0.45:
            rec = "MAINTENANCE ADVISORY: Elevated mechanical and thermal strain observed. Schedule diagnostic inspection for cooling efficiency and fuel regulator within 500 KM."
            risk_level = "WARNING"
        elif risk_score > 0.25:
            rec = "ATTENTION: Minor variance detected across secondary sensors. Systems operational, recommend preventative inspection at next routine service interval."
            risk_level = "MODERATE"
        else:
            rec = "SYSTEM NOMINAL: All 8 powertrain sensors operating inside factory-certified engineering tolerance envelopes. Powertrain health certified."
            risk_level = "NOMINAL"

        return {
            'risk_scores': [risk_score],
            'failure_probabilities': [failure_prob],
            'anomaly_scores': [anomaly_score],
            'recommendations': [rec],
            'risk_level': risk_level,
            'subsystem_issues': subsystem_issues
        }


model = VehiclePredictiveMaintenanceModel()

# -----------------------------------------------------------------------------
# SESSION STATE & PRESET CALIBRATION PROFILES
# -----------------------------------------------------------------------------
def set_scenario(rpm, temp, load, speed, throttle, fuel, air, km):
    st.session_state.engine_rpm = rpm
    st.session_state.coolant_temp = temp
    st.session_state.engine_load = load
    st.session_state.vehicle_speed = speed
    st.session_state.throttle_pos = throttle
    st.session_state.fuel_pressure = fuel
    st.session_state.air_temp = air
    st.session_state.mileage = km

if "engine_rpm" not in st.session_state:
    set_scenario(4200, 94, 52, 186, 64, 45, 26, 48000)

if "unit_mode" not in st.session_state:
    st.session_state.unit_mode = "KM/H"

# -----------------------------------------------------------------------------
# SIDEBAR: ENGINEERING TELEMETRY CONTROLS (Zero Emojis)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.08);">
        <div>
            <div style="font-weight:800; font-size:16px; letter-spacing:0.04em; color:#FFF;">VECTIS // COCKPIT</div>
            <div style="font-size:10px; color:#8E9BAE; letter-spacing:0.08em; text-transform:uppercase;">Telematics Calibration</div>
        </div>
        <span style="font-size:9px; font-weight:700; background:rgba(34,197,94,0.15); color:#22C55E; padding:3px 8px; border-radius:4px; border:1px solid rgba(34,197,94,0.4); letter-spacing:0.08em;">ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px; font-weight:700; color:#8E9BAE; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>DRIVE PROFILES</div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("CRUISE", use_container_width=True):
            set_scenario(2400, 90, 36, 110, 30, 46, 24, 42000)
            st.rerun()
        if st.button("THERMAL SOAK", use_container_width=True):
            set_scenario(5400, 118, 76, 160, 70, 38, 38, 140000)
            st.rerun()
    with col_p2:
        if st.button("PERFORMANCE", use_container_width=True):
            set_scenario(6200, 96, 68, 220, 68, 50, 26, 48000)
            st.rerun()
        if st.button("CRITICAL FAULT", use_container_width=True):
            set_scenario(7600, 132, 94, 280, 96, 24, 44, 230000)
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; font-weight:700; color:#8E9BAE; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>POWERTRAIN CONTROLS</div>", unsafe_allow_html=True)

    engine_rpm = st.slider("Engine Speed (RPM)", 800, 8500, value=int(st.session_state.engine_rpm), step=50, key="engine_rpm")
    engine_load = st.slider("Manifold Mechanical Load (%)", 0, 100, value=int(st.session_state.engine_load), step=1, key="engine_load")
    throttle_pos = st.slider("Throttle Valve Angle (%)", 0, 100, value=int(st.session_state.throttle_pos), step=1, key="throttle_pos")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; font-weight:700; color:#8E9BAE; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>THERMAL & INJECTION</div>", unsafe_allow_html=True)

    coolant_temp = st.slider("Coolant Temperature (°C)", 40, 145, value=int(st.session_state.coolant_temp), step=1, key="coolant_temp")
    fuel_pressure = st.slider("Common Rail Fuel Pressure (PSI)", 15, 80, value=int(st.session_state.fuel_pressure), step=1, key="fuel_pressure")
    air_temp = st.slider("Intake Air Temperature (°C)", -15, 55, value=int(st.session_state.air_temp), step=1, key="air_temp")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; font-weight:700; color:#8E9BAE; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>VELOCITY & ODOMETER</div>", unsafe_allow_html=True)

    vehicle_speed = st.slider("Ground Velocity (KM/H)", 0, 350, value=int(st.session_state.vehicle_speed), step=1, key="vehicle_speed")
    mileage = st.number_input("Cumulative Mileage (KM)", 1000, 300000, value=int(st.session_state.mileage), step=1000, key="mileage")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("SYNCHRONIZE TELEMETRY", type="primary", use_container_width=True):
        st.rerun()


# Package inputs & predict
input_data = {
    'Engine_RPM': engine_rpm,
    'Coolant_Temperature': coolant_temp,
    'Engine_Load': engine_load,
    'Vehicle_Speed': vehicle_speed,
    'Throttle_Position': throttle_pos,
    'Fuel_Pressure': fuel_pressure,
    'Air_Temperature': air_temp,
    'Mileage': mileage
}

results = model.predict_maintenance_needs(input_data)
risk_score = results['risk_scores'][0]
failure_prob = results['failure_probabilities'][0]
anomaly_score = results['anomaly_scores'][0]
recommendation = results['recommendations'][0]
risk_level = results['risk_level']
subsystem_issues = results['subsystem_issues']


# -----------------------------------------------------------------------------
# TOP COMMAND BAR (Minimalist, Engineering Aesthetic)
# -----------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; padding:12px 18px; margin-bottom:20px; background:#11141B; border:1px solid rgba(255,255,255,0.08); border-radius:14px;">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-weight:800; font-size:18px; letter-spacing:0.06em; color:#FFF;">
            SPEED<span style="color:#22C55E;">X</span> // <span style="font-weight:500; font-size:14px; color:#8E9BAE;">TELEMETRY SYSTEM</span>
        </div>
        <div class="speedx-pill" style="border-color:rgba(34,197,94,0.3); background:rgba(34,197,94,0.08); color:#22C55E;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#22C55E;"></span>
            STREAM SYNCHRONIZED
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:12px; font-family:'JetBrains Mono', monospace; font-size:11px; color:#8E9BAE;">
        <span>CAN-BUS: <strong style="color:#FFF;">1000 KBPS</strong></span>
        <span>&bull;</span>
        <span>LATENCY: <strong style="color:#22C55E;">2.4 MS</strong></span>
        <span>&bull;</span>
        <span>NODE: <strong style="color:#FFF;">PRIMARY ECU</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MASTER COCKPIT GRID (3 Columns)
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([1.05, 1.35, 1.1])

# =============================================================================
# LEFT COLUMN: VEHICLE CHASSIS & DISTANCE TELEMETRY
# =============================================================================
with col1:
    # Card 1: Sports Car Showcase Card
    st.markdown(f"""
    <div class="speedx-card">
        <div class="speedx-card-title">
            <span>CHASSIS PROFILE // SPEC-04</span>
            <span style="font-family:'JetBrains Mono', monospace; color:#22C55E; font-size:10px;">CAN-ID: 0x7E0</span>
        </div>
        <div style="text-align:center; margin:10px 0; position:relative; overflow:hidden; border-radius:10px; background:rgba(0,0,0,0.2);">
            <img src="data:image/png;base64,{CAR_B64}" style="width:100%; height:auto; display:block; border-radius:8px;" />
        </div>
        <div style="display:flex; align-items:center; justify-content:space-between; margin-top:12px; font-size:11px; font-family:'JetBrains Mono', monospace;">
            <span style="font-weight:700; color:#FFFFFF;">4.0L V8 TWIN-TURBO</span>
            <span class="speedx-pill" style="padding:2px 8px; font-size:10px; background:rgba(34,197,94,0.12); border-color:rgba(34,197,94,0.3); color:#22C55E;">
                CHASSIS NOMINAL
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-cards: Odometer & Structural Integrity Segments
    sub_col_a, sub_col_b = st.columns(2)
    with sub_col_a:
        dist_pct = min(int((mileage / 150000.0) * 100), 100)
        st.markdown(f"""
        <div class="speedx-card" style="padding:14px;">
            <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                ODOMETER
            </div>
            <div style="font-size:20px; font-weight:800; color:#FFF; margin:6px 0 2px 0; font-family:'JetBrains Mono', monospace;">
                {mileage / 1000.0:.1f}K <span style="font-size:11px; color:#8E9BAE;">KM</span>
            </div>
            <div class="speedx-bar-container">
                <div class="speedx-bar-fill" style="width:{dist_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sub_col_b:
        num_lit = max(1, min(5, int((1.0 - risk_score) * 5) + 1))
        seg_html = "".join([f'<div class="speedx-segment-block {"lit" if i < num_lit else ""}"></div>' for i in range(5)])
        st.markdown(f"""
        <div class="speedx-card" style="padding:14px;">
            <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                INTEGRITY INDEX
            </div>
            <div style="font-size:20px; font-weight:800; color:#FFF; margin:6px 0 2px 0; font-family:'JetBrains Mono', monospace;">
                {num_lit} / 5 <span style="font-size:11px; color:#8E9BAE;">CELLS</span>
            </div>
            <div class="speedx-segments">
                {seg_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Card 3: Next Service Horizon Indicator
    rem_service = max(0, 10000 - (mileage % 10000))
    st.markdown(f"""
    <div class="speedx-card" style="padding:14px; margin-bottom:0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                    MAINTENANCE HORIZON
                </div>
                <div style="font-size:16px; font-weight:800; color:#FFF; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                    {rem_service:,} KM <span style="font-size:11px; color:#8E9BAE; font-weight:500;">UNTIL SERVICE</span>
                </div>
            </div>
            <span class="speedx-pill" style="font-size:10px; border-color:rgba(34,197,94,0.3); color:#22C55E;">SCHEDULED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CENTER COLUMN: SPEEDOMETER DIAL & CORE POWERTRAIN TELEMETRY
# (Guaranteed 100% Reliable Render via st.components.v1.html)
# =============================================================================
with col2:
    # Needle angle formula: -130 deg (0 km/h) to +130 deg (350 km/h)
    needle_deg = -130.0 + (min(vehicle_speed, 350) / 350.0) * 260.0
    fuel_pct = min(100, int((fuel_pressure / 70.0) * 100))

    # Construct bulletproof HTML/SVG component
    dial_component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: transparent;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: #FFFFFF;
        overflow: hidden;
      }}
      .speedx-card-inner {{
        background: #11141B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 14px 18px 10px 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
      }}
    </style>
    </head>
    <body>
      <div class="speedx-card-inner">
        <!-- Top Fuel Rail Indicator Bar -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; color:#8E9BAE; font-weight:700;">
          <span style="display:flex; align-items:center; gap:6px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#22C55E;"></span>
            COMMON RAIL FUEL INJECTION
          </span>
          <span style="font-family:'JetBrains Mono', monospace; color:#F59E0B; font-weight:700;">{fuel_pressure} PSI</span>
        </div>
        <div style="background:rgba(255,255,255,0.06); height:6px; border-radius:3px; overflow:hidden; margin-bottom:10px;">
          <div style="background:linear-gradient(90deg, #22C55E 0%, #F59E0B 75%, #EF4444 100%); height:100%; width:{fuel_pct}%;"></div>
        </div>

        <!-- Speedometer Dial SVG -->
        <div style="position:relative; width:100%; max-width:320px; margin:0 auto; height:185px;">
          <svg viewBox="0 0 340 185" style="width:100%; height:100%; overflow:visible;">
            <defs>
              <linearGradient id="speedArcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#EF4444" />
                <stop offset="25%" stop-color="#F59E0B" />
                <stop offset="65%" stop-color="#22C55E" />
                <stop offset="100%" stop-color="#10B981" />
              </linearGradient>
            </defs>

            <!-- Dial Arc Background Track -->
            <path d="M 40,150 A 130,130 0 1,1 300,150" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="12" stroke-linecap="round"/>
            <path d="M 40,150 A 130,130 0 1,1 300,150" fill="none" stroke="url(#speedArcGrad)" stroke-width="5" stroke-linecap="round" stroke-dasharray="3 6"/>

            <!-- Speed Markings -->
            <text x="48" y="150" fill="#64748B" font-size="11" font-weight="700">0</text>
            <text x="65" y="95" fill="#64748B" font-size="11" font-weight="700">50</text>
            <text x="115" y="50" fill="#CBD5E1" font-size="11" font-weight="700">150</text>
            <text x="170" y="34" fill="#CBD5E1" font-size="11" font-weight="700" text-anchor="middle">200</text>
            <text x="235" y="95" fill="#22C55E" font-size="11" font-weight="700">280</text>
            <text x="285" y="150" fill="#22C55E" font-size="11" font-weight="700">350</text>

            <!-- Digital Velocity Readout -->
            <text x="170" y="118" fill="#FFFFFF" font-size="48" font-weight="800" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif">{vehicle_speed}</text>
            <text x="170" y="140" fill="#8E9BAE" font-size="11" font-weight="700" text-anchor="middle" letter-spacing="1.5">KM / H</text>

            <!-- Crimson Baseline Bar -->
            <line x1="60" y1="162" x2="280" y2="162" stroke="#EF4444" stroke-width="3" stroke-linecap="round"/>

            <!-- Precision Rotating Needle -->
            <g transform="translate(170, 150)">
              <g transform="rotate({needle_deg})">
                <line x1="0" y1="0" x2="0" y2="-105" stroke="#22C55E" stroke-width="3" stroke-linecap="round"/>
                <polygon points="-3,-105 0,-115 3,-105" fill="#4ADE80"/>
                <circle cx="0" cy="0" r="7" fill="#11141B" stroke="#22C55E" stroke-width="2.5"/>
              </g>
            </g>
          </svg>
        </div>
      </div>
    </body>
    </html>
    """

    components.html(dial_component_html, height=265, scrolling=False)

    # Sub-metrics below Gauge (Zero Emojis, Clean Engineering Layout)
    row_sub1, row_sub2 = st.columns(2)
    with row_sub1:
        st.markdown(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:12px;">
            <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                MANIFOLD LOAD
            </div>
            <div style="font-size:18px; font-weight:800; color:#FFF; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                {engine_load}% <span style="font-size:10px; color:#8E9BAE; font-weight:500;">TORQUE RATIO</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        dmg_pct = int(risk_score * 100)
        c_dmg = "#EF4444" if dmg_pct > 65 else ("#F59E0B" if dmg_pct > 35 else "#22C55E")
        st.markdown(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:0;">
            <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                DEGRADATION RISK
            </div>
            <div style="font-size:20px; font-weight:800; color:{c_dmg}; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                {dmg_pct}% <span style="font-size:10px; color:#8E9BAE; font-weight:500;">INDEX</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row_sub2:
        st.markdown(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:12px;">
            <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                THROTTLE ANGLE
            </div>
            <div style="font-size:18px; font-weight:800; color:#FFF; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                {throttle_pos}% <span style="font-size:10px; color:#8E9BAE; font-weight:500;">BUTTERFLY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_temp = "#EF4444" if coolant_temp > 115 else ("#F59E0B" if coolant_temp > 100 else "#22C55E")
        st.markdown(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:0;">
            <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">
                COOLANT THERMAL
            </div>
            <div style="font-size:20px; font-weight:800; color:{c_temp}; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                {coolant_temp}°C <span style="font-size:10px; color:#8E9BAE; font-weight:500;">BLOCK</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# RIGHT COLUMN: SUBSYSTEM DIAGNOSTIC RADAR & TELEMATICS COMMS
# =============================================================================
with col3:
    # Card 1: System Health / Failure Probability
    fail_pct = int(failure_prob * 100)
    c_fail = "#EF4444" if fail_pct > 60 else ("#F59E0B" if fail_pct > 30 else "#22C55E")

    st.markdown(f"""
    <div class="speedx-card" style="padding-bottom:14px;">
        <div class="speedx-card-title">
            <span>PREDICTIVE HEALTH INDEX</span>
            <span style="color:{c_fail}; font-family:'JetBrains Mono', monospace; font-size:10px;">{risk_level}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div style="font-size:28px; font-weight:800; color:{c_fail}; font-family:'JetBrains Mono', monospace;">
                {fail_pct}%
            </div>
            <div style="text-align:right; font-size:11px; color:#8E9BAE; font-family:'JetBrains Mono', monospace;">
                ANOMALY SCORE: <strong style="color:#FFF;">{anomaly_score:.2f}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Interactive Subsystem Health Radar (Replacing video-game screenshot)
    # Calculate live stress scores (0-100) for 5 diagnostic axes
    thermal_stress = min(100, max(15, int((coolant_temp - 60) / 70.0 * 100)))
    load_stress = min(100, max(15, int(engine_load)))
    fuel_stress = min(100, max(15, int(abs(fuel_pressure - 45) / 30.0 * 100 + 20)))
    rpm_stress = min(100, max(15, int((engine_rpm - 1000) / 7000.0 * 100)))
    chassis_stress = min(100, max(15, int((mileage / 200000.0) * 50 + (vehicle_speed / 350.0) * 50)))

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[thermal_stress, load_stress, fuel_stress, rpm_stress, chassis_stress],
        theta=['Thermal', 'Manifold', 'Fuel Rail', 'Crankshaft', 'Driveline'],
        fill='toself',
        fillcolor='rgba(34, 197, 94, 0.22)',
        line=dict(color='#22C55E', width=2),
        marker=dict(size=4, color='#4ADE80'),
        hoverinfo='theta+r'
    ))

    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.08)',
                linecolor='rgba(255, 255, 255, 0.08)'
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#8E9BAE', family='Plus Jakarta Sans'),
                gridcolor='rgba(255, 255, 255, 0.08)',
                linecolor='rgba(255, 255, 255, 0.08)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=20, b=20),
        height=190,
        showlegend=False
    )

    st.markdown("""
    <div class="speedx-card" style="padding:14px 16px 6px 16px;">
        <div class="speedx-card-title">
            <span>SUBSYSTEM HEALTH RADAR</span>
            <span style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#22C55E;">5-AXIS VECTOR</span>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Card 3: Telematics AI Advisory (Clean, Authentic Engineering Feedback)
    st.markdown(f"""
    <div class="speedx-card" style="margin-bottom:0;">
        <div class="speedx-card-title">
            <span>DIAGNOSTIC ADVISORY // ECU TELEMETRY</span>
        </div>
        <div class="speedx-quote-box">
            {recommendation}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; font-size:10px; font-family:'JetBrains Mono', monospace; color:#8E9BAE;">
            <span>ACTIVE FAULT CODES: <strong style="color:#FFF;">{len(subsystem_issues)}</strong></span>
            <span style="color:#22C55E; font-weight:700;">TELEMETRY STREAM VERIFIED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DETAILED SPECIFICATION TABS (Preserving Sensor Matrix & Diagnostic Codes)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

tab_matrix, tab_dtc, tab_decomp = st.tabs([
    "SENSOR MATRIX & FACTORY TOLERANCES",
    "ON-BOARD DIAGNOSTIC CODES (OBD-II DTC)",
    "FAILURE PROBABILITY & ANOMALY DECOMPOSITION"
])

with tab_matrix:
    def classify_sensor(p, v):
        if p == "Engine_RPM":
            return ("CRITICAL", "red") if v > 7200 else (("WARN", "amber") if v > 6000 else ("NOMINAL", "green"))
        elif p == "Coolant_Temperature":
            return ("CRITICAL", "red") if v > 115 else (("WARN", "amber") if v > 102 else ("NOMINAL", "green"))
        elif p == "Engine_Load":
            return ("CRITICAL", "red") if v > 85 else (("WARN", "amber") if v > 75 else ("NOMINAL", "green"))
        elif p == "Fuel_Pressure":
            return ("CRITICAL", "red") if (v < 34 or v > 66) else (("WARN", "amber") if (v < 38 or v > 60) else ("NOMINAL", "green"))
        elif p == "Mileage":
            return ("WARN", "amber") if v > 160000 else ("NOMINAL", "green")
        else:
            return ("NOMINAL", "green")

    matrix_data = [
        ("Engine_RPM", engine_rpm, "1,500 – 6,000", "RPM", "Crankshaft rotational angular frequency"),
        ("Coolant_Temperature", coolant_temp, "85 – 102", "°C", "Primary cooling circuit thermal state"),
        ("Engine_Load", engine_load, "20 – 75", "%", "Normalized cylinder charge and torque demand"),
        ("Vehicle_Speed", vehicle_speed, "0 – 250", "KM/H", "Wheel hub velocity from ABS wheel sensors"),
        ("Throttle_Position", throttle_pos, "10 – 80", "%", "Drive-by-wire electronic throttle blade angle"),
        ("Fuel_Pressure", fuel_pressure, "38 – 60", "PSI", "Common rail high-pressure injection loop"),
        ("Air_Temperature", air_temp, "10 – 40", "°C", "Intake manifold charge air temperature"),
        ("Mileage", mileage, "0 – 160,000", "KM", "Vehicle lifecycle cumulative distance")
    ]

    rows_html = ""
    for p_name, p_val, p_opt, p_unit, p_desc in matrix_data:
        s_tag, s_color = classify_sensor(p_name, p_val)
        hex_color = "#22C55E" if s_color == "green" else ("#F59E0B" if s_color == "amber" else "#EF4444")
        val_repr = f"{p_val:,}" if isinstance(p_val, int) else f"{p_val}"
        rows_html += f'<tr><td style="font-weight:700; color:#FFFFFF;">{p_name}</td><td style="color:{hex_color}; font-weight:700;">{val_repr}</td><td style="color:var(--text-muted);">{p_opt}</td><td style="color:var(--text-dim);">{p_unit}</td><td style="color:var(--text-muted);">{p_desc}</td><td><span style="background:rgba(255,255,255,0.06); color:{hex_color}; padding:3px 8px; border-radius:4px; font-size:10px; font-weight:700;">{s_tag}</span></td></tr>'

    st.markdown(f"""
    <div class="speedx-card">
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>SENSOR TELEMETRY</th>
                    <th>LIVE READING</th>
                    <th>CERTIFIED SPEC</th>
                    <th>UNIT</th>
                    <th>SUBSYSTEM DESCRIPTION</th>
                    <th>STATUS</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

with tab_dtc:
    dtc_records = []
    if coolant_temp > 115:
        dtc_records.append(("P0217", "Engine Coolant Over Temperature Condition", "Cooling Loop", "CRITICAL", "Inspect radiator fan relay and cooling fluid flow immediately."))
    elif coolant_temp > 102:
        dtc_records.append(("P0117", "Engine Coolant Temp Sensor Circuit Low", "Sensors", "WARNING", "Inspect coolant wiring harness resistance and thermostatic valve."))

    if fuel_pressure < 34:
        dtc_records.append(("P0087", "Fuel Rail/System Pressure Too Low", "Fuel Injection", "CRITICAL", "High-pressure fuel pump delivery deficit or blocked fuel filter."))
    elif fuel_pressure > 65:
        dtc_records.append(("P0088", "Fuel Rail/System Pressure Too High", "Fuel Delivery", "WARNING", "Fuel pressure regulator stuck closed or return line restricted."))

    if engine_rpm > 7200:
        dtc_records.append(("P0219", "Engine Overspeed Condition", "Powertrain", "CRITICAL", "Crankshaft rotational speed exceeded rev-limiter safety threshold."))

    if engine_load > 88:
        dtc_records.append(("P0299", "Turbocharger / Supercharger Underboost / High Load Deficit", "Induction", "WARNING", "Excessive manifold pressure demand detected."))

    if not dtc_records:
        dtc_records.append(("P0000", "No Active Fault Codes in ECU Memory", "Powertrain", "NOMINAL", "All sensor metrics within certified factory operational envelopes."))

    dtc_rows = ""
    for code, desc, sys_name, sev, action in dtc_records:
        c_sev = "#22C55E" if sev == "NOMINAL" else ("#F59E0B" if sev == "WARNING" else "#EF4444")
        dtc_rows += f'<tr><td style="font-family:\'JetBrains Mono\'; font-weight:700; color:{c_sev};">{code}</td><td style="font-weight:600; color:#FFF;">{desc}</td><td style="color:var(--text-muted);">{sys_name}</td><td><span style="color:{c_sev}; font-weight:700;">{sev}</span></td><td style="color:var(--text-dim);">{action}</td></tr>'

    st.markdown(f"""
    <div class="speedx-card">
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>DTC CODE</th>
                    <th>FAULT DESCRIPTION</th>
                    <th>SUBSYSTEM</th>
                    <th>SEVERITY</th>
                    <th>WORKSHOP DIAGNOSTIC PROCEDURE</th>
                </tr>
            </thead>
            <tbody>
                {dtc_rows}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

with tab_decomp:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        # Failure Probability Distribution
        fig_prob = go.Figure(go.Indicator(
            mode="gauge+number",
            value=failure_prob * 100,
            number={'suffix': "%", 'font': {'color': "#FFFFFF", 'family': 'JetBrains Mono', 'size': 36}},
            title={'text': "PREDICTIVE FAILURE PROBABILITY", 'font': {'color': "#8E9BAE", 'size': 12, 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#8E9BAE", 'tickfont': {'family': 'JetBrains Mono', 'size': 10}},
                'bar': {'color': "#22C55E" if failure_prob < 0.3 else ("#F59E0B" if failure_prob < 0.6 else "#EF4444")},
                'bgcolor': "rgba(255,255,255,0.06)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.15)"},
                    {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                ]
            }
        ))
        fig_prob.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.markdown('<div class="speedx-card" style="padding:14px;">', unsafe_allow_html=True)
        st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d2:
        # Anomaly score gauge
        fig_anom = go.Figure(go.Indicator(
            mode="gauge+number",
            value=anomaly_score * 100,
            number={'suffix': "%", 'font': {'color': "#FFFFFF", 'family': 'JetBrains Mono', 'size': 36}},
            title={'text': "MULTIVARIATE ANOMALY DEVIATION", 'font': {'color': "#8E9BAE", 'size': 12, 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#8E9BAE", 'tickfont': {'family': 'JetBrains Mono', 'size': 10}},
                'bar': {'color': "#22C55E" if anomaly_score < 0.35 else ("#F59E0B" if anomaly_score < 0.7 else "#EF4444")},
                'bgcolor': "rgba(255,255,255,0.06)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': "rgba(34, 197, 94, 0.15)"},
                    {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                ]
            }
        ))
        fig_anom.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.markdown('<div class="speedx-card" style="padding:14px;">', unsafe_allow_html=True)
        st.plotly_chart(fig_anom, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; margin-top:28px; margin-bottom:12px; font-size:10px; color:#64748B; font-family:'JetBrains Mono', monospace; letter-spacing:0.1em;">
    VECTIS PREDICTIVE TELEMETRICS &bull; ON-BOARD DIAGNOSTIC CAN-BUS TELEMATICS ENGINE &bull; VERSION 4.1.0
</div>
""", unsafe_allow_html=True)
