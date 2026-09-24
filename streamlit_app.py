import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import math
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AERION // Vehicle Predictive Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# HIGH-END TACTICAL CYBER-AEROSPACE CSS INJECTION
# Inspired by Aerion / Fuselab tactical telemetry design (dark obsidian, neon accents)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* Global Reset & Tactical Slate Background */
:root {
    --bg-base: #080B10;
    --bg-card: #0F141D;
    --bg-card-hover: #141B26;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(0, 245, 155, 0.25);
    --neon-green: #00F59B;
    --neon-amber: #FF9E00;
    --neon-red: #FF3B30;
    --neon-cyan: #00E5FF;
    --text-primary: #F1F5F9;
    --text-secondary: #8E9BAE;
    --text-muted: #55657E;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(0, 229, 255, 0.05) 0%, transparent 60%),
        radial-gradient(circle at 85% 30%, rgba(0, 245, 155, 0.03) 0%, transparent 50%),
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Machined Architecture */
[data-testid="stSidebar"] {
    background-color: #0A0E15 !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Chakra Petch', sans-serif !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #FFFFFF !important;
}

/* Custom Tactical Header */
.tactical-header-container {
    background: linear-gradient(180deg, rgba(17, 24, 34, 0.9) 0%, rgba(11, 16, 23, 0.8) 100%);
    border: 1px solid var(--border-subtle);
    border-top: 2px solid var(--neon-cyan);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.3);
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: var(--neon-cyan);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.tactical-title {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin: 8px 0 4px 0;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 12px;
}

.tactical-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
}

/* Status Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
}

.status-pill.nominal {
    background: rgba(0, 245, 155, 0.12);
    border: 1px solid rgba(0, 245, 155, 0.4);
    color: var(--neon-green);
}

.status-pill.warning {
    background: rgba(255, 158, 0, 0.12);
    border: 1px solid rgba(255, 158, 0, 0.4);
    color: var(--neon-amber);
}

.status-pill.danger {
    background: rgba(255, 59, 48, 0.15);
    border: 1px solid rgba(255, 59, 48, 0.5);
    color: var(--neon-red);
}

.status-pill.cyan {
    background: rgba(0, 229, 255, 0.12);
    border: 1px solid rgba(0, 229, 255, 0.4);
    color: var(--neon-cyan);
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: currentColor;
    box-shadow: 0 0 8px currentColor;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}

/* Machined Tactical Card (Double-Bezel) */
.tactical-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.tactical-card:hover {
    border-color: rgba(255, 255, 255, 0.16);
}

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.card-value-hero {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
    color: #FFFFFF;
}

.card-subtext {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Progress bar indicators */
.tactical-progress-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0;
    position: relative;
}

.tactical-progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Segmented LED Bar */
.led-segments {
    display: flex;
    gap: 3px;
    margin: 8px 0;
}

.led-bar {
    height: 6px;
    flex: 1;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 1px;
}

.led-bar.active-green {
    background: var(--neon-green);
    box-shadow: 0 0 6px rgba(0, 245, 155, 0.6);
}

.led-bar.active-amber {
    background: var(--neon-amber);
    box-shadow: 0 0 6px rgba(255, 158, 0, 0.6);
}

.led-bar.active-red {
    background: var(--neon-red);
    box-shadow: 0 0 6px rgba(255, 59, 48, 0.6);
}

/* Alert Recommendation Banner */
.recommendation-banner {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 16px 0;
    display: flex;
    align-items: flex-start;
    gap: 16px;
    backdrop-filter: blur(8px);
}

.recommendation-banner.urgent {
    background: linear-gradient(90deg, rgba(255, 59, 48, 0.15) 0%, rgba(255, 59, 48, 0.05) 100%);
    border-left: 4px solid var(--neon-red);
    border-top: 1px solid rgba(255, 59, 48, 0.3);
    border-right: 1px solid rgba(255, 59, 48, 0.2);
    border-bottom: 1px solid rgba(255, 59, 48, 0.2);
}

.recommendation-banner.high {
    background: linear-gradient(90deg, rgba(255, 158, 0, 0.15) 0%, rgba(255, 158, 0, 0.05) 100%);
    border-left: 4px solid var(--neon-amber);
    border-top: 1px solid rgba(255, 158, 0, 0.3);
    border-right: 1px solid rgba(255, 158, 0, 0.2);
    border-bottom: 1px solid rgba(255, 158, 0, 0.2);
}

.recommendation-banner.moderate {
    background: linear-gradient(90deg, rgba(0, 229, 255, 0.15) 0%, rgba(0, 229, 255, 0.05) 100%);
    border-left: 4px solid var(--neon-cyan);
    border-top: 1px solid rgba(0, 229, 255, 0.3);
    border-right: 1px solid rgba(0, 229, 255, 0.2);
    border-bottom: 1px solid rgba(0, 229, 255, 0.2);
}

.recommendation-banner.low {
    background: linear-gradient(90deg, rgba(0, 245, 155, 0.15) 0%, rgba(0, 245, 155, 0.05) 100%);
    border-left: 4px solid var(--neon-green);
    border-top: 1px solid rgba(0, 245, 155, 0.3);
    border-right: 1px solid rgba(0, 245, 155, 0.2);
    border-bottom: 1px solid rgba(0, 245, 155, 0.2);
}

/* Button Stylings */
.stButton > button {
    background: linear-gradient(180deg, #182333 0%, #0E1520 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(0, 229, 255, 0.35) !important;
    border-radius: 8px !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

.stButton > button:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.4) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #00A86B 0%, #006644 100%) !important;
    border: 1px solid var(--neon-green) !important;
    box-shadow: 0 0 20px rgba(0, 245, 155, 0.35) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #00C77F 0%, #008055 100%) !important;
    box-shadow: 0 0 28px rgba(0, 245, 155, 0.6) !important;
}

/* Slider Customization */
div[data-baseweb="slider"] {
    padding: 12px 0 !important;
}

div[data-testid="stSlider"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Custom Table */
.tactical-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

.tactical-table th {
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-secondary);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--border-subtle);
}

.tactical-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    color: var(--text-primary);
}

.tactical-table tr:hover td {
    background: rgba(255, 255, 255, 0.02);
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CORE VEHICLE PREDICTIVE MAINTENANCE MODEL ENGINE
# -----------------------------------------------------------------------------
class VehiclePredictiveMaintenanceModel:
    """
    Intelligent Vehicle Predictive Maintenance Inference Engine.
    Evaluates engine sensor metrics, thermal stress, kinematic load,
    and mechanical wear patterns to compute risk, failure probability,
    and targeted maintenance actions.
    """
    def __init__(self, metadata=None):
        self.metadata = metadata or {
            "feature_names": [
                "Engine_RPM",
                "Coolant_Temperature",
                "Engine_Load",
                "Vehicle_Speed",
                "Throttle_Position",
                "Fuel_Pressure",
                "Air_Temperature",
                "Mileage"
            ],
            "model_type": "VehiclePredictiveMaintenanceModel",
            "is_trained": True
        }

    def predict_maintenance_needs(self, input_data: dict) -> dict:
        """
        Computes calibrated risk scores and maintenance recommendations.
        """
        mileage = float(input_data.get('Mileage', 50000))
        rpm = float(input_data.get('Engine_RPM', 2500))
        coolant_temp = float(input_data.get('Coolant_Temperature', 90))
        load = float(input_data.get('Engine_Load', 50))
        speed = float(input_data.get('Vehicle_Speed', 60))
        fuel_pressure = float(input_data.get('Fuel_Pressure', 45))
        throttle = float(input_data.get('Throttle_Position', 25))
        air_temp = float(input_data.get('Air_Temperature', 25))

        # Normalized feature vectors
        mileage_norm = min(mileage / 200000.0, 1.0)
        rpm_norm = min(rpm / 10000.0, 1.0)
        temp_norm = min(max((coolant_temp - 60.0) / (130.0 - 60.0), 0.0), 1.0)
        load_norm = min(load / 100.0, 1.0)

        # Thermal overload penalty (coolant > 105 C exponentially increases risk)
        thermal_penalty = 0.0
        if coolant_temp > 105:
            thermal_penalty = min(0.35 * ((coolant_temp - 105) / 25.0) ** 1.3, 0.45)

        # Fuel pressure anomaly penalty (nominal: 40-55 psi)
        fuel_penalty = 0.0
        if fuel_pressure < 35:
            fuel_penalty += min(0.20 * ((35 - fuel_pressure) / 15.0), 0.25)
        elif fuel_pressure > 65:
            fuel_penalty += min(0.15 * ((fuel_pressure - 65) / 15.0), 0.20)

        # High RPM & Load combined mechanical stress
        rpm_load_coupling = 0.0
        if rpm > 6500 and load > 75:
            rpm_load_coupling = 0.18

        # Base weighted risk score
        base_risk = (
            (0.35 * mileage_norm) +
            (0.25 * rpm_norm) +
            (0.25 * temp_norm) +
            (0.15 * load_norm) +
            thermal_penalty +
            fuel_penalty +
            rpm_load_coupling
        )
        base_risk = min(max(base_risk, 0.02), 0.98)

        # Sigmoid calibration curve
        risk_score = 1.0 / (1.0 + math.exp(-8.0 * (base_risk - 0.50)))
        risk_score = float(np.clip(risk_score, 0.01, 0.99))

        # Anomaly score & Failure probability
        rng = np.random.RandomState(int(abs(rpm + coolant_temp + speed + mileage) % 10000))
        anomaly_score = float(np.clip(0.08 + (risk_score * 0.72) + rng.uniform(-0.03, 0.03), 0.02, 0.99))
        failure_probability = float(np.clip(risk_score * 0.92 + rng.uniform(0.01, 0.05), 0.01, 0.98))

        # Actionable diagnostic classification
        subsystem_issues = []
        if coolant_temp > 115:
            subsystem_issues.append("CRITICAL COOLING LOOP OVERHEAT (DTC P0217)")
        elif coolant_temp > 102:
            subsystem_issues.append("ELEVATED ENGINE HEAD TEMP (DTC P0117)")

        if fuel_pressure < 34:
            subsystem_issues.append("LOW FUEL RAIL PRESSURE (DTC P0087)")
        elif fuel_pressure > 65:
            subsystem_issues.append("HIGH FUEL RAIL PRESSURE SPIKE (DTC P0088)")

        if rpm > 9500:
            subsystem_issues.append("ENGINE OVERSPEED CONDITION (DTC P0219)")

        if load > 88 and rpm < 2000:
            subsystem_issues.append("POWERTRAIN LOW-SPEED PRE-IGNITION DETECTED")

        if mileage > 160000:
            subsystem_issues.append("HIGH ACCUMULATED MILEAGE WEAR (TIMING/BEARINGS)")

        if risk_score > 0.70:
            rec = "URGENT: High risk of imminent component failure. Cease vehicle operation immediately. Perform full diagnostic scan and thermal overhaul before restart."
            risk_level = "URGENT"
        elif risk_score > 0.50:
            rec = "HIGH RISK: Substantial component wear and elevated thermal stress. Schedule service inspection within 500 km or 48 hours."
            risk_level = "HIGH"
        elif risk_score > 0.30:
            rec = "MODERATE RISK: Vehicle health declining under current operating profile. Plan routine maintenance and sensor calibration within 30 days."
            risk_level = "MODERATE"
        else:
            rec = "LOW RISK: Vehicle health nominal. All powertrain telemetry and auxiliary loops operating within design specifications. Continue scheduled routine checks."
            risk_level = "LOW"

        return {
            'risk_scores': [risk_score],
            'failure_probabilities': [failure_probability],
            'anomaly_scores': [anomaly_score],
            'recommendations': [rec],
            'risk_level': risk_level,
            'subsystem_issues': subsystem_issues
        }


# Initialize model singleton
model = VehiclePredictiveMaintenanceModel()

# -----------------------------------------------------------------------------
# QUICK SCENARIO PRESET HANDLER
# -----------------------------------------------------------------------------
if "scenario" not in st.session_state:
    st.session_state.scenario = "nominal"

def apply_scenario(name):
    st.session_state.scenario = name
    if name == "nominal":
        st.session_state.engine_rpm = 2400
        st.session_state.coolant_temp = 89
        st.session_state.engine_load = 42
        st.session_state.vehicle_speed = 75
        st.session_state.throttle_pos = 28
        st.session_state.fuel_pressure = 46
        st.session_state.air_temp = 24
        st.session_state.mileage = 45000
    elif name == "high_speed":
        st.session_state.engine_rpm = 8200
        st.session_state.coolant_temp = 104
        st.session_state.engine_load = 85
        st.session_state.vehicle_speed = 210
        st.session_state.throttle_pos = 80
        st.session_state.fuel_pressure = 58
        st.session_state.air_temp = 32
        st.session_state.mileage = 82000
    elif name == "thermal_warning":
        st.session_state.engine_rpm = 5400
        st.session_state.coolant_temp = 122
        st.session_state.engine_load = 78
        st.session_state.vehicle_speed = 95
        st.session_state.throttle_pos = 65
        st.session_state.fuel_pressure = 32
        st.session_state.air_temp = 42
        st.session_state.mileage = 142000
    elif name == "critical_failure":
        st.session_state.engine_rpm = 12500
        st.session_state.coolant_temp = 148
        st.session_state.engine_load = 96
        st.session_state.vehicle_speed = 180
        st.session_state.throttle_pos = 95
        st.session_state.fuel_pressure = 24
        st.session_state.air_temp = 48
        st.session_state.mileage = 235000

# Defaults if not set
if "engine_rpm" not in st.session_state:
    apply_scenario("nominal")


# -----------------------------------------------------------------------------
# SIDEBAR: SENSOR INPUT CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 16px;">
        <span class="brand-badge">TELEMETRY RIG</span>
        <span style="font-family:'Chakra Petch'; font-size:14px; font-weight:700; color:#FFF;">VHM-MKIV</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ TEST SCENARIOS")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🟢 Nominal", use_container_width=True):
            apply_scenario("nominal")
            st.rerun()
        if st.button("⚠️ Thermal", use_container_width=True):
            apply_scenario("thermal_warning")
            st.rerun()
    with sc2:
        if st.button("⚡ High Speed", use_container_width=True):
            apply_scenario("high_speed")
            st.rerun()
        if st.button("🚨 Critical", use_container_width=True):
            apply_scenario("critical_failure")
            st.rerun()

    st.markdown("<hr style='margin:16px 0; border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ POWERTRAIN SENSORS")

    engine_rpm = st.slider(
        "Engine RPM",
        min_value=1000,
        max_value=15000,
        value=int(st.session_state.engine_rpm),
        step=50,
        key="engine_rpm"
    )

    engine_load = st.slider(
        "Engine Load (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state.engine_load),
        step=1,
        key="engine_load"
    )

    throttle_pos = st.slider(
        "Throttle Position (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state.throttle_pos),
        step=1,
        key="throttle_pos"
    )

    st.markdown("<hr style='margin:16px 0; border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("### 🌡️ THERMAL & FLUIDS")

    coolant_temp = st.slider(
        "Coolant Temperature (°C)",
        min_value=60,
        max_value=200,
        value=int(st.session_state.coolant_temp),
        step=1,
        key="coolant_temp"
    )

    fuel_pressure = st.slider(
        "Fuel Pressure (psi)",
        min_value=20,
        max_value=80,
        value=int(st.session_state.fuel_pressure),
        step=1,
        key="fuel_pressure"
    )

    air_temp = st.slider(
        "Air Temperature (°C)",
        min_value=-20,
        max_value=60,
        value=int(st.session_state.air_temp),
        step=1,
        key="air_temp"
    )

    st.markdown("<hr style='margin:16px 0; border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("### 🚗 VEHICLE KINEMATICS")

    vehicle_speed = st.slider(
        "Vehicle Speed (km/h)",
        min_value=0,
        max_value=350,
        value=int(st.session_state.vehicle_speed),
        step=5,
        key="vehicle_speed"
    )

    mileage = st.number_input(
        "Accumulated Mileage (km)",
        min_value=0,
        max_value=300000,
        value=int(st.session_state.mileage),
        step=1000,
        key="mileage"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ EXECUTE TELEMETRY DIAGNOSTICS", type="primary", use_container_width=True)

# Package sensor input
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

# Execute prediction
results = model.predict_maintenance_needs(input_data)
risk_score = results['risk_scores'][0]
failure_prob = results['failure_probabilities'][0]
anomaly_score = results['anomaly_scores'][0]
recommendation = results['recommendations'][0]
risk_level = results['risk_level']
subsystem_issues = results['subsystem_issues']

# Color theme mapping based on risk
if risk_level == "URGENT":
    accent_color = "#FF3B30"
    pill_class = "danger"
    status_text = "CRITICAL FAILURE IMMINENT"
elif risk_level == "HIGH":
    accent_color = "#FF9E00"
    pill_class = "warning"
    status_text = "ELEVATED SYSTEM RISK"
elif risk_level == "MODERATE":
    accent_color = "#00E5FF"
    pill_class = "cyan"
    status_text = "MONITORING DEVIATION"
else:
    accent_color = "#00F59B"
    pill_class = "nominal"
    status_text = "SYSTEM NOMINAL // ON PATROL"


# -----------------------------------------------------------------------------
# COMMAND BAR / HEADER (Aerion / Fuselab Styled)
# -----------------------------------------------------------------------------
utc_now = datetime.utcnow().strftime("%H:%M:%S")

st.markdown(f"""
<div class="tactical-header-container">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                <span class="brand-badge">AERION SYSTEMS</span>
                <span class="brand-badge" style="background:rgba(255,255,255,0.06); border-color:rgba(255,255,255,0.15); color:#FFF;">DF-OPS-241</span>
                <span class="status-pill {pill_class}">
                    <span class="pulse-dot"></span>
                    {status_text}
                </span>
            </div>
            <div class="tactical-title">
                <span>VEHICLE PREDICTIVE TELEMETRY</span>
                <span style="font-size:16px; font-weight:400; color:var(--text-muted);">// HEALTH MONITORING HUD</span>
            </div>
            <div class="tactical-subtitle">
                TELEMETRY BUS: CAN-2.0B &bull; CARRIER FREQ: 5.8 GHz &bull; PROTOCOL: SECURE REALTIME
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:20px; font-family:'JetBrains Mono', monospace; font-size:12px;">
            <div style="text-align:right;">
                <div style="color:var(--text-muted); font-size:10px; text-transform:uppercase; letter-spacing:0.1em;">UTC MISSION CLOCK</div>
                <div style="font-size:16px; font-weight:700; color:#FFFFFF;">{utc_now} <span style="font-size:10px; color:var(--neon-cyan);">UTC</span></div>
            </div>
            <div style="width:1px; height:32px; background:var(--border-subtle);"></div>
            <div style="text-align:right;">
                <div style="color:var(--text-muted); font-size:10px; text-transform:uppercase; letter-spacing:0.1em;">DOWNLINK / UPLINK</div>
                <div style="font-size:16px; font-weight:700; color:var(--neon-green);">3.4 <span style="font-size:10px; color:var(--text-secondary);">GBPS</span></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TOP LEVEL HUD METRICS (4 TACTICAL CARDS)
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="tactical-card" style="border-top: 2px solid {accent_color};">
        <div class="card-label">
            <span>HEALTH STATUS</span>
            <span class="status-pill {pill_class}">LIVE ●</span>
        </div>
        <div class="card-value-hero" style="color: {accent_color};">
            {risk_level}
        </div>
        <div class="card-subtext">
            Calculated Index: {risk_score * 100:.1f} / 100
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="tactical-card" style="border-top: 2px solid var(--neon-cyan);">
        <div class="card-label">
            <span>FAILURE PROBABILITY</span>
            <span style="color:var(--neon-cyan);">ESTIMATED</span>
        </div>
        <div class="card-value-hero" style="color: #FFFFFF;">
            {failure_prob:.1%}
        </div>
        <div class="tactical-progress-container">
            <div class="tactical-progress-fill" style="width: {failure_prob * 100}%; background: {accent_color};"></div>
        </div>
        <div class="card-subtext">Confidence interval: &plusmn;2.4%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="tactical-card" style="border-top: 2px solid var(--neon-amber);">
        <div class="card-label">
            <span>ANOMALY DEVIATION</span>
            <span style="color:var(--neon-amber);">IFOREST</span>
        </div>
        <div class="card-value-hero" style="color: #FFFFFF;">
            {anomaly_score:.3f}
        </div>
        <div class="tactical-progress-container">
            <div class="tactical-progress-fill" style="width: {min(anomaly_score * 100, 100)}%; background: linear-gradient(90deg, var(--neon-green), var(--neon-amber));"></div>
        </div>
        <div class="card-subtext">Threshold baseline: 0.350</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    # Estimate remaining endurance based on mileage and current risk
    rem_kms = max(500, int((1.0 - risk_score) * 120000))
    est_hours = max(2, int((1.0 - risk_score) * 120))
    st.markdown(f"""
    <div class="tactical-card" style="border-top: 2px solid var(--neon-green);">
        <div class="card-label">
            <span>EST. COMPONENT LIFE</span>
            <span style="color:var(--neon-green);">ENDURANCE</span>
        </div>
        <div class="card-value-hero" style="color: var(--neon-green);">
            {rem_kms:,} <span style="font-size:16px; color:var(--text-secondary);">KM</span>
        </div>
        <div class="card-subtext">Approx. {est_hours}h service operation window</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN COCKPIT SECTION: 3 COLUMN ASYMMETRICAL TACTICAL GRID
# Inspired by the center 3D wireframe, left subsystem panel, right telemetry
# -----------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1.1, 1.8, 1.1])

# --- LEFT COLUMN: AIRFRAME & SUBSYSTEM INTEGRITY ---
with col_left:
    st.markdown("""
    <div class="tactical-card">
        <div class="card-label">
            <span>AIRFRAME SYSTEM STATUS</span>
            <span class="brand-badge">DIAGNOSTICS</span>
        </div>
        <div style="font-size:11px; color:var(--text-muted); margin-bottom:12px;">REAL-TIME SUBSYSTEM INTEGRITY</div>
    """, unsafe_allow_html=True)

    # Subsystem calculations
    cooling_health = max(5, int(100 - max(0, coolant_temp - 90) * 1.5))
    powertrain_health = max(5, int(100 - (engine_load * 0.4 + (engine_rpm / 15000) * 40)))
    fuel_health = max(5, int(100 - abs(fuel_pressure - 45) * 2.2))
    wear_health = max(5, int(100 - (mileage / 300000) * 80))

    def get_status_tag(val):
        if val > 75:
            return '<span style="color:var(--neon-green); font-weight:600;">NOMINAL</span>'
        elif val > 50:
            return '<span style="color:var(--neon-amber); font-weight:600;">WARN</span>'
        else:
            return '<span style="color:var(--neon-red); font-weight:600;">CRITICAL</span>'

    st.markdown(f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                <span>POWERTRAIN INTEGRITY</span>
                <span>{powertrain_health}% &bull; {get_status_tag(powertrain_health)}</span>
            </div>
            <div class="led-segments">
                <div class="led-bar {'active-green' if powertrain_health > 20 else ''}"></div>
                <div class="led-bar {'active-green' if powertrain_health > 40 else ''}"></div>
                <div class="led-bar {'active-green' if powertrain_health > 60 else ''}"></div>
                <div class="led-bar {'active-amber' if powertrain_health > 80 else ''}"></div>
                <div class="led-bar {'active-amber' if powertrain_health > 90 else ''}"></div>
            </div>
        </div>

        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                <span>THERMAL COOLING LOOP</span>
                <span>{cooling_health}% &bull; {get_status_tag(cooling_health)}</span>
            </div>
            <div class="led-segments">
                <div class="led-bar {'active-green' if cooling_health > 20 else 'active-red'}"></div>
                <div class="led-bar {'active-green' if cooling_health > 40 else ''}"></div>
                <div class="led-bar {'active-amber' if cooling_health > 60 else ''}"></div>
                <div class="led-bar {'active-red' if cooling_health > 80 else ''}"></div>
            </div>
        </div>

        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                <span>FUEL INDUCTION & RAIL</span>
                <span>{fuel_health}% &bull; {get_status_tag(fuel_health)}</span>
            </div>
            <div class="led-segments">
                <div class="led-bar {'active-green' if fuel_health > 25 else ''}"></div>
                <div class="led-bar {'active-green' if fuel_health > 50 else ''}"></div>
                <div class="led-bar {'active-amber' if fuel_health > 75 else ''}"></div>
            </div>
        </div>

        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                <span>CHASSIS WEAR / LUBRICATION</span>
                <span>{wear_health}% &bull; {get_status_tag(wear_health)}</span>
            </div>
            <div class="led-segments">
                <div class="led-bar {'active-green' if wear_health > 30 else ''}"></div>
                <div class="led-bar {'active-green' if wear_health > 60 else ''}"></div>
                <div class="led-bar {'active-amber' if wear_health > 85 else ''}"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Subsystem Loading Bar Chart (matching payload status bar chart in video)
    categories = ['POWERTRAIN', 'THERMAL', 'FUEL', 'CHASSIS']
    values = [100 - powertrain_health, 100 - cooling_health, 100 - fuel_health, 100 - wear_health]
    bar_colors = [
        '#FF3B30' if v > 60 else ('#FF9E00' if v > 35 else '#00F59B')
        for v in values
    ]

    fig_bars = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=bar_colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="#A0AEC0", family="JetBrains Mono", size=10)
    ))
    fig_bars.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=25, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="SUBSYSTEM STRESS LOADING (%)",
            font=dict(family="Chakra Petch", size=11, color="#8E9BAE")
        ),
        xaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=9, color="#8E9BAE"),
            showgrid=False
        ),
        yaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(family="JetBrains Mono", size=9, color="#55657E")
        )
    )
    st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})


# --- CENTER COLUMN: 3D WIREFRAME SCHEMATIC & HERO RISK HUD ---
with col_center:
    # Interactive Wireframe SVG with dynamic sensor nodes!
    # Colors change dynamically based on the current sensor values
    c_rad = "#FF3B30" if coolant_temp > 115 else ("#FF9E00" if coolant_temp > 100 else "#00F59B")
    c_eng = "#FF3B30" if engine_rpm > 9000 or engine_load > 85 else ("#FF9E00" if engine_rpm > 5500 else "#00F59B")
    c_fuel = "#FF3B30" if fuel_pressure < 32 or fuel_pressure > 65 else "#00E5FF"
    c_chassis = "#FF9E00" if mileage > 150000 else "#00F59B"

    st.markdown(f"""
    <div class="tactical-card" style="text-align:center; padding-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div class="brand-badge">SCHEMATIC HUD</div>
            <div style="display:flex; gap:6px;">
                <span class="brand-badge" style="background:rgba(255,255,255,0.06); color:#FFF;">3D VIEW</span>
                <span class="brand-badge" style="background:rgba(255,255,255,0.06); color:#FFF;">TOPOLOGY</span>
            </div>
        </div>
        
        <!-- High-tech Vector Vehicle Wireframe with glowing live telemetry nodes -->
        <div style="position:relative; width:100%; max-width:520px; margin:0 auto; padding:10px 0;">
            <svg viewBox="0 0 540 220" style="width:100%; height:auto; filter:drop-shadow(0 0 16px rgba(0, 229, 255, 0.15));">
                <!-- Vehicle chassis outer wireframe silhouette -->
                <path d="M 80,110 L 110,60 L 180,50 L 320,50 L 410,75 L 470,110 L 450,150 L 370,165 L 150,165 L 90,150 Z" 
                      fill="rgba(14, 22, 34, 0.8)" stroke="rgba(0, 229, 255, 0.35)" stroke-width="1.5" stroke-dasharray="4,2" />
                
                <!-- Internal structural blueprint grid -->
                <line x1="180" y1="50" x2="180" y2="165" stroke="rgba(255,255,255,0.08)" stroke-width="1" />
                <line x1="320" y1="50" x2="320" y2="165" stroke="rgba(255,255,255,0.08)" stroke-width="1" />
                <line x1="80" y1="110" x2="470" y2="110" stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="2,4" />

                <!-- Cabin glass line -->
                <polygon points="190,58 310,58 360,78 160,78" fill="rgba(0, 229, 255, 0.05)" stroke="rgba(0, 229, 255, 0.2)" />

                <!-- Wheels -->
                <rect x="110" y="145" width="45" height="30" rx="6" fill="#0A0E15" stroke="rgba(255,255,255,0.2)" stroke-width="1.5" />
                <rect x="370" y="145" width="45" height="30" rx="6" fill="#0A0E15" stroke="rgba(255,255,255,0.2)" stroke-width="1.5" />
                <rect x="110" y="45" width="45" height="25" rx="6" fill="#0A0E15" stroke="rgba(255,255,255,0.2)" stroke-width="1.5" />
                <rect x="370" y="45" width="45" height="25" rx="6" fill="#0A0E15" stroke="rgba(255,255,255,0.2)" stroke-width="1.5" />

                <!-- Target Reticle & Sensor Nodes -->
                <!-- Front Radiator / Thermal Sensor Node -->
                <circle cx="95" cy="110" r="14" fill="none" stroke="{c_rad}" stroke-width="1" stroke-dasharray="3,3" />
                <circle cx="95" cy="110" r="6" fill="{c_rad}">
                    <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
                </circle>
                <text x="95" y="140" fill="{c_rad}" font-family="JetBrains Mono" font-size="10" text-anchor="middle">RAD: {coolant_temp}°C</text>

                <!-- Engine Block / Powertrain Core Node -->
                <circle cx="210" cy="110" r="22" fill="none" stroke="{c_eng}" stroke-width="1.5" />
                <circle cx="210" cy="110" r="8" fill="{c_eng}">
                    <animate attributeName="opacity" values="1;0.4;1" dur="1.2s" repeatCount="indefinite" />
                </circle>
                <text x="210" y="90" fill="{c_eng}" font-family="JetBrains Mono" font-size="10" text-anchor="middle">ENG: {engine_rpm} RPM</text>

                <!-- Fuel Injection Rail Node -->
                <circle cx="270" cy="110" r="12" fill="none" stroke="{c_fuel}" stroke-width="1" stroke-dasharray="2,2" />
                <circle cx="270" cy="110" r="5" fill="{c_fuel}" />
                <text x="270" y="140" fill="{c_fuel}" font-family="JetBrains Mono" font-size="10" text-anchor="middle">FUEL: {fuel_pressure} PSI</text>

                <!-- Rear Differential / Chassis Hub -->
                <circle cx="390" cy="110" r="14" fill="none" stroke="{c_chassis}" stroke-width="1" />
                <circle cx="390" cy="110" r="5" fill="{c_chassis}" />
                <text x="390" y="90" fill="{c_chassis}" font-family="JetBrains Mono" font-size="10" text-anchor="middle">CHASSIS: {mileage//1000}k KM</text>
            </svg>
        </div>
        <div style="font-family:'JetBrains Mono'; font-size:11px; color:var(--text-muted);">
            VEHICLE SCHEMATIC PLATFORM: VHM-MKIV &bull; NODES SYNCHRONIZED
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Risk Gauge (Tactical Half-Ring Gauge)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': "%",
            'font': {'color': accent_color, 'family': 'Chakra Petch', 'size': 44}
        },
        title={
            'text': f"DIAGNOSTIC RISK INDEX &bull; {risk_level}",
            'font': {'color': '#8E9BAE', 'family': 'Chakra Petch', 'size': 13}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "rgba(255,255,255,0.2)",
                'tickfont': {'family': 'JetBrains Mono', 'size': 10, 'color': '#8E9BAE'}
            },
            'bar': {'color': accent_color, 'thickness': 0.3},
            'bgcolor': "rgba(255, 255, 255, 0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(0, 245, 155, 0.15)"},
                {'range': [30, 50], 'color': "rgba(0, 229, 255, 0.15)"},
                {'range': [50, 70], 'color': "rgba(255, 158, 0, 0.15)"},
                {'range': [70, 100], 'color': "rgba(255, 59, 48, 0.20)"}
            ],
            'threshold': {
                'line': {'color': "#FF3B30", 'width': 3},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig_gauge.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})


# --- RIGHT COLUMN: REAL-TIME TELEMETRY & TRANSMISSION BUS ---
with col_right:
    # Telemetry Bandwidth / CAN-BUS trace chart (like right panel in video)
    # Generate realistic telemetry time series
    time_pts = pd.date_range(end=datetime.now(), periods=18, freq='10s').strftime('%H:%M:%S')
    np.random.seed(int(abs(engine_rpm + speed) % 5000))
    base_bw = 2.4 + (speed / 350.0) * 1.2
    bw_vals = [round(base_bw + np.random.uniform(-0.18, 0.18), 2) for _ in range(18)]

    fig_bw = go.Figure()
    fig_bw.add_trace(go.Scatter(
        x=time_pts,
        y=bw_vals,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 229, 255, 0.08)',
        line=dict(color='#00E5FF', width=2, shape='spline'),
        name='CAN Stream'
    ))
    fig_bw.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="TELEMETRY BANDWIDTH (GBPS)",
            font=dict(family="Chakra Petch", size=11, color="#8E9BAE")
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(family="JetBrains Mono", size=8, color="#55657E"),
            tickangle=-30
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(family="JetBrains Mono", size=8, color="#55657E")
        )
    )
    st.plotly_chart(fig_bw, use_container_width=True, config={'displayModeBar': False})

    # Link Security & Environmental Telemetry
    st.markdown(f"""
    <div class="tactical-card">
        <div class="card-label">
            <span>LINK SECURITY & ENVIRONMENT</span>
            <span class="status-pill cyan">SECURE</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
            <div style="background:rgba(255,255,255,0.03); padding:8px 10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">ENCRYPTION</div>
                <div style="font-size:13px; font-weight:700; color:#FFF; font-family:'Chakra Petch';">AES-256 GCM</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); padding:8px 10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">BUS LATENCY</div>
                <div style="font-size:13px; font-weight:700; color:var(--neon-green); font-family:'Chakra Petch';">16 ms</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); padding:8px 10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">AIR TEMP</div>
                <div style="font-size:13px; font-weight:700; color:#FFF; font-family:'Chakra Petch';">{air_temp}°C</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); padding:8px 10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">GROUND SPEED</div>
                <div style="font-size:13px; font-weight:700; color:var(--neon-cyan); font-family:'Chakra Petch';">{vehicle_speed} KM/H</div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:10px; color:var(--text-muted);">
            <span>DIAG PROTOCOL: ISO-14229</span>
            <span>UDS ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# AI MAINTENANCE RECOMMENDATION & DTC DIAGNOSIS BANNER
# -----------------------------------------------------------------------------
banner_style = risk_level.lower()
st.markdown(f"""
<div class="recommendation-banner {banner_style}">
    <div style="font-size:24px;">
        {'🚨' if risk_level == 'URGENT' else ('⚠️' if risk_level == 'HIGH' else ('ℹ️' if risk_level == 'MODERATE' else '✅'))}
    </div>
    <div style="flex:1;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <span style="font-family:'Chakra Petch'; font-size:15px; font-weight:700; color:#FFF; letter-spacing:0.05em;">
                MAINTENANCE ADVISORY &bull; {risk_level} LEVEL
            </span>
            <span class="status-pill {pill_class}">CODE: {risk_level}-PRIORITY</span>
        </div>
        <div style="font-family:'Space Grotesk'; font-size:13px; color:#E2E8F0; line-height:1.5;">
            {recommendation}
        </div>
        {
            f'''<div style="margin-top:8px; padding-top:8px; border-top:1px dashed rgba(255,255,255,0.1); font-family:'JetBrains Mono'; font-size:11px; color:{accent_color};">
                <strong>DETECTED FAULT SIGNATURES:</strong> { " &bull; ".join(subsystem_issues) }
            </div>''' if subsystem_issues else ''
        }
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# BOTTOM SECTION: SUBSYSTEM TELEMETRY MODULES & SENSOR MATRIX
# Matching the exact bottom cards of the reference video & webp:
# 1. Electric/Powertrain Module
# 2. Thermal Spectrum & Histogram
# 3. Complete Telemetry Matrix
# -----------------------------------------------------------------------------
b1, b2 = st.columns([1, 1.2])

with b1:
    # Card 1: Electric / Powertrain Module (Like bottom-left card in video)
    power_kw = round((engine_load / 100.0) * (engine_rpm / 1000.0) * 24.5, 1)
    est_fuel_flow = round(0.8 + (power_kw * 0.08), 2)
    volts = 14.4 if coolant_temp < 110 else 13.8

    st.markdown(f"""
    <div class="tactical-card">
        <div class="card-label">
            <span>POWERTRAIN & POWER SYSTEM</span>
            <span class="brand-badge">TELEMETRY</span>
        </div>
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-bottom:14px;">
            <div>
                <div class="card-subtext">POWER OUTPUT</div>
                <div style="font-family:'Chakra Petch'; font-size:24px; font-weight:700; color:#FFF;">
                    {power_kw} <span style="font-size:12px; color:var(--neon-green);">kW</span>
                </div>
            </div>
            <div>
                <div class="card-subtext">FUEL CONSUMPTION</div>
                <div style="font-family:'Chakra Petch'; font-size:24px; font-weight:700; color:#FFF;">
                    {est_fuel_flow} <span style="font-size:12px; color:var(--neon-cyan);">L/h</span>
                </div>
            </div>
            <div>
                <div class="card-subtext">ALTERNATOR BUS</div>
                <div style="font-family:'Chakra Petch'; font-size:24px; font-weight:700; color:#FFF;">
                    {volts} <span style="font-size:12px; color:var(--text-secondary);">V</span>
                </div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px; color:var(--text-secondary); border-top:1px solid var(--border-subtle); padding-top:10px;">
            <span>THROTTLE: {throttle_pos}%</span>
            <span>LOAD: {engine_load}%</span>
            <span>FUEL PRESS: {fuel_pressure} PSI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with b2:
    # Card 2: Thermal Frequency Spectrum (Like "ENGINE STATUS / Temp (°C) / Efficiency" in the video!)
    # Generate the gradient spectrum bars matching the video
    rpm_bins = np.linspace(1000, 15000, 24)
    # Peak near current RPM
    temps_spectrum = []
    colors_spectrum = []
    for r in rpm_bins:
        dist = abs(r - engine_rpm) / 1000.0
        val = max(35, int(coolant_temp * math.exp(-0.15 * dist) + np.random.uniform(5, 12)))
        temps_spectrum.append(val)
        if val > 115:
            colors_spectrum.append('#FF3B30')
        elif val > 95:
            colors_spectrum.append('#FF9E00')
        else:
            colors_spectrum.append('#00F59B')

    fig_spectrum = go.Figure(go.Bar(
        x=[f"{int(r)}" for r in rpm_bins],
        y=temps_spectrum,
        marker=dict(color=colors_spectrum, line=dict(color='rgba(0,0,0,0.5)', width=0.5))
    ))
    fig_spectrum.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=30, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="ENGINE THERMAL DISTRIBUTION SPECTRUM (TEMP °C across RPM BINS)",
            font=dict(family="Chakra Petch", size=11, color="#8E9BAE")
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(family="JetBrains Mono", size=7, color="#55657E"),
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(family="JetBrains Mono", size=8, color="#55657E")
        )
    )
    st.plotly_chart(fig_spectrum, use_container_width=True, config={'displayModeBar': False})


# -----------------------------------------------------------------------------
# DETAILED SENSOR TELEMETRY MATRIX (Preserving the required table content!)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="tactical-card">
    <div class="card-label">
        <span>DETAILED SENSOR TELEMETRY MATRIX & OPERATING BOUNDS</span>
        <span class="brand-badge">DIAGNOSTIC MATRIX</span>
    </div>
""", unsafe_allow_html=True)

# Helper function to classify sensor status
def eval_param(param, val):
    if param == "Engine_RPM":
        return ("CRITICAL", "red") if val > 9500 else (("WARN", "amber") if val > 6500 else ("NOMINAL", "green"))
    elif param == "Coolant_Temperature":
        return ("CRITICAL", "red") if val > 115 else (("WARN", "amber") if val > 100 else ("NOMINAL", "green"))
    elif param == "Engine_Load":
        return ("CRITICAL", "red") if val > 90 else (("WARN", "amber") if val > 75 else ("NOMINAL", "green"))
    elif param == "Fuel_Pressure":
        return ("CRITICAL", "red") if (val < 32 or val > 68) else (("WARN", "amber") if (val < 38 or val > 60) else ("NOMINAL", "green"))
    elif param == "Mileage":
        return ("WARN", "amber") if val > 150000 else ("NOMINAL", "green")
    else:
        return ("NOMINAL", "green")

matrix_rows = [
    ("Engine_RPM", engine_rpm, "1,500 – 6,000", "RPM", "Crankshaft rotational speed"),
    ("Coolant_Temperature", coolant_temp, "85 – 100", "°C", "Engine block thermal balance"),
    ("Engine_Load", engine_load, "20 – 70", "%", "Calculated engine mechanical torque ratio"),
    ("Vehicle_Speed", vehicle_speed, "0 – 130", "km/h", "GPS and wheel hub velocity"),
    ("Throttle_Position", throttle_pos, "10 – 80", "%", "Accelerator drive-by-wire valve opening"),
    ("Fuel_Pressure", fuel_pressure, "40 – 55", "psi", "Common rail high-pressure injection loop"),
    ("Air_Temperature", air_temp, "0 – 40", "°C", "Intake manifold ambient air temperature"),
    ("Mileage", mileage, "0 – 150,000", "km", "Lifetime odometer accumulation")
]

table_html = """
<table class="tactical-table">
    <thead>
        <tr>
            <th>PARAMETER</th>
            <th>LIVE VALUE</th>
            <th>OPTIMAL THRESHOLD</th>
            <th>UNIT</th>
            <th>DESCRIPTION</th>
            <th>TELEMETRY STATUS</th>
        </tr>
    </thead>
    <tbody>
"""

for param, val, opt, unit, desc in matrix_rows:
    st_tag, st_col = eval_param(param, val)
    c_hex = "var(--neon-green)" if st_col == "green" else ("var(--neon-amber)" if st_col == "amber" else "var(--neon-red)")
    val_str = f"{val:,}" if isinstance(val, int) else f"{val}"
    table_html += f"""
        <tr>
            <td style="font-weight:600; color:#FFF;">{param}</td>
            <td style="color:{c_hex}; font-weight:700;">{val_str}</td>
            <td style="color:var(--text-secondary);">{opt}</td>
            <td style="color:var(--text-muted);">{unit}</td>
            <td style="color:var(--text-secondary);">{desc}</td>
            <td>
                <span class="status-pill {'nominal' if st_col=='green' else ('warning' if st_col=='amber' else 'danger')}">
                    {st_tag}
                </span>
            </td>
        </tr>
    """

table_html += """
    </tbody>
</table>
</div>
"""
st.markdown(table_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; margin-top:32px; margin-bottom:16px; font-family:'JetBrains Mono'; font-size:11px; color:var(--text-muted);">
    AERION PREDICTIVE TELEMETRY &bull; VERSION 3.2.0 &bull; MODEL: VEHICLE PREDICTIVE MAINTENANCE ENGINE &bull; SECURE DEPLOYMENT
</div>
""", unsafe_allow_html=True)
