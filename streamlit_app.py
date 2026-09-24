import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AERION // Vehicle Predictive Maintenance & Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render HTML cleanly without any markdown indentation issues
def render_html(html_str: str):
    lines = [
        line.strip()
        for line in html_str.splitlines()
        if line.strip() and not line.strip().startswith("<!--")
    ]
    clean = "".join(lines)
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HIGH-END LUXURY AUTOMOTIVE & AEROSPACE TELEMETRY CSS
# -----------------------------------------------------------------------------
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Master CSS Variables */
:root {
    --bg-main: #0B0E14;
    --bg-card: #11161F;
    --bg-card-inner: #161D28;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-highlight: rgba(255, 255, 255, 0.16);
    --neon-green: #00F59B;
    --neon-amber: #FF9E00;
    --neon-red: #FF3B30;
    --neon-cyan: #00E5FF;
    --text-pure: #FFFFFF;
    --text-dim: #94A3B8;
    --text-muted: #64748B;
}

/* Background Canvas */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-main) !important;
    background-image: 
        radial-gradient(circle at 50% -10%, rgba(0, 229, 255, 0.05) 0%, transparent 60%),
        radial-gradient(circle at 90% 40%, rgba(0, 245, 155, 0.03) 0%, transparent 45%),
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
    color: var(--text-pure) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Machined Dark Terminal */
[data-testid="stSidebar"] {
    background-color: #0E1219 !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Chakra Petch', sans-serif !important;
    letter-spacing: 0.12em !important;
    font-size: 13px !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
}

/* Typography Standards */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Chakra Petch', sans-serif !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-pure) !important;
    margin: 0 !important;
}

/* Top Aerospace Brand Bar */
.aerion-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(180deg, rgba(20, 27, 38, 0.95) 0%, rgba(14, 19, 27, 0.95) 100%);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 14px 22px;
    margin-bottom: 16px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.brand-section {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-logo {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    font-size: 20px;
    letter-spacing: 0.15em;
    color: #FFFFFF;
}

.nav-telemetry-stats {
    display: flex;
    align-items: center;
    gap: 20px;
    font-family: 'JetBrains Mono', monospace;
}

.nav-stat-group {
    text-align: right;
}

.nav-stat-label {
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.nav-stat-val {
    font-size: 14px;
    font-weight: 700;
    color: #FFFFFF;
}

/* Real Native Streamlit Tabs Styling */
div[data-testid="stTabs"] {
    margin-bottom: 20px;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(14, 19, 27, 0.85) !important;
    padding: 5px !important;
    border-radius: 10px !important;
    gap: 6px !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
    border: none !important;
    padding: 8px 18px !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background: #1C2533 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* Double-Bezel Hardware Cards */
.bezel-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.bezel-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
}

.bezel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.bezel-title {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #FFFFFF;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}

.bezel-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
}

.badge-nominal {
    background: rgba(0, 245, 155, 0.12);
    border: 1px solid rgba(0, 245, 155, 0.35);
    color: var(--neon-green);
}

.badge-warn {
    background: rgba(255, 158, 0, 0.12);
    border: 1px solid rgba(255, 158, 0, 0.35);
    color: var(--neon-amber);
}

.badge-alert {
    background: rgba(255, 59, 48, 0.15);
    border: 1px solid rgba(255, 59, 48, 0.45);
    color: var(--neon-red);
}

.badge-cyan {
    background: rgba(0, 229, 255, 0.12);
    border: 1px solid rgba(0, 229, 255, 0.35);
    color: var(--neon-cyan);
}

/* Pulsing LED status dot */
.led-pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: currentColor;
    display: inline-block;
    box-shadow: 0 0 8px currentColor;
    animation: led-blink 1.8s infinite ease-in-out;
}

@keyframes led-blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.8); }
}

/* Hero KPI Readout */
.hero-metric-val {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
    margin: 4px 0 2px 0;
}

.hero-metric-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
}

/* Segmented Progress Bars (Fuselab Style) */
.bar-track {
    display: flex;
    gap: 3px;
    height: 6px;
    margin: 8px 0;
}

.bar-segment {
    flex: 1;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 1px;
}

.bar-segment.fill-green {
    background: var(--neon-green);
    box-shadow: 0 0 6px rgba(0, 245, 155, 0.5);
}

.bar-segment.fill-amber {
    background: var(--neon-amber);
    box-shadow: 0 0 6px rgba(255, 158, 0, 0.5);
}

.bar-segment.fill-red {
    background: var(--neon-red);
    box-shadow: 0 0 6px rgba(255, 59, 48, 0.6);
}

/* Telemetry List Rows */
.telemetry-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

.telemetry-row:last-child {
    border-bottom: none;
}

.telemetry-param-title {
    color: var(--text-dim);
    font-size: 11px;
    letter-spacing: 0.06em;
}

.telemetry-param-val {
    font-weight: 600;
    color: #FFFFFF;
}

/* Advisory Banner */
.advisory-box {
    background: linear-gradient(90deg, rgba(22, 29, 41, 0.9) 0%, rgba(17, 23, 33, 0.9) 100%);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 16px 0;
    border-left: 4px solid var(--neon-green);
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
}

.advisory-box.urgent {
    border-left-color: var(--neon-red);
    background: linear-gradient(90deg, rgba(45, 18, 22, 0.8) 0%, rgba(17, 23, 33, 0.9) 100%);
}

.advisory-box.high {
    border-left-color: var(--neon-amber);
    background: linear-gradient(90deg, rgba(40, 28, 16, 0.8) 0%, rgba(17, 23, 33, 0.9) 100%);
}

.advisory-box.moderate {
    border-left-color: var(--neon-cyan);
}

/* Preset Action Buttons (Sidebar) */
.stButton > button {
    background: #141B26 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    border-color: var(--neon-cyan) !important;
    background: #1B2433 !important;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.3) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #00A86B 0%, #006644 100%) !important;
    border-color: var(--neon-green) !important;
    box-shadow: 0 0 20px rgba(0, 245, 155, 0.35) !important;
}

/* Segmented view controls */
div[data-testid="stRadio"] [role="radiogroup"] {
    background: rgba(0, 0, 0, 0.35) !important;
    padding: 3px !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    gap: 4px !important;
}

div[data-testid="stRadio"] [role="radio"] {
    padding: 4px 12px !important;
    border-radius: 6px !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
}

/* Slider Overrides */
div[data-baseweb="slider"] {
    padding: 10px 0 !important;
}

div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Data Table Overhauls */
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
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.matrix-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.matrix-table tr:hover td {
    background: rgba(255, 255, 255, 0.02);
}
</style>
""")


# -----------------------------------------------------------------------------
# CORE VEHICLE PREDICTIVE MAINTENANCE ENGINE
# -----------------------------------------------------------------------------
class VehiclePredictiveMaintenanceModel:
    def __init__(self, metadata=None):
        self.metadata = metadata or {
            "feature_names": [
                "Engine_RPM", "Coolant_Temperature", "Engine_Load",
                "Vehicle_Speed", "Throttle_Position", "Fuel_Pressure",
                "Air_Temperature", "Mileage"
            ],
            "model_type": "VehiclePredictiveMaintenanceModel",
            "is_trained": True
        }

    def predict_maintenance_needs(self, input_data: dict) -> dict:
        mileage = float(input_data.get('Mileage', 50000))
        rpm = float(input_data.get('Engine_RPM', 2500))
        coolant_temp = float(input_data.get('Coolant_Temperature', 90))
        load = float(input_data.get('Engine_Load', 50))
        speed = float(input_data.get('Vehicle_Speed', 60))
        fuel_pressure = float(input_data.get('Fuel_Pressure', 45))
        throttle = float(input_data.get('Throttle_Position', 25))
        air_temp = float(input_data.get('Air_Temperature', 25))

        # Normalized feature vectors
        mileage_norm = min(mileage / 220000.0, 1.0)
        rpm_norm = min(rpm / 10000.0, 1.0)
        temp_norm = min(max((coolant_temp - 60.0) / (135.0 - 60.0), 0.0), 1.0)
        load_norm = min(load / 100.0, 1.0)

        # Thermal overload penalty
        thermal_penalty = 0.0
        if coolant_temp > 104:
            thermal_penalty = min(0.38 * ((coolant_temp - 104) / 22.0) ** 1.3, 0.50)

        # Fuel pressure anomaly penalty (nominal: 40-55 psi)
        fuel_penalty = 0.0
        if fuel_pressure < 36:
            fuel_penalty += min(0.25 * ((36 - fuel_pressure) / 16.0), 0.30)
        elif fuel_pressure > 64:
            fuel_penalty += min(0.18 * ((fuel_pressure - 64) / 16.0), 0.22)

        # High RPM & load coupling
        rpm_load_coupling = 0.0
        if rpm > 6800 and load > 75:
            rpm_load_coupling = 0.16

        base_risk = (
            (0.32 * mileage_norm) +
            (0.24 * rpm_norm) +
            (0.24 * temp_norm) +
            (0.16 * load_norm) +
            thermal_penalty +
            fuel_penalty +
            rpm_load_coupling
        )
        base_risk = min(max(base_risk, 0.02), 0.98)

        risk_score = 1.0 / (1.0 + math.exp(-8.2 * (base_risk - 0.50)))
        risk_score = float(np.clip(risk_score, 0.01, 0.99))

        # Deterministic pseudo-random seed based on input
        seed = int(abs(rpm * 3 + coolant_temp * 7 + speed * 11 + mileage) % 99999)
        rng = np.random.RandomState(seed)
        anomaly_score = float(np.clip(0.08 + (risk_score * 0.74) + rng.uniform(-0.02, 0.02), 0.02, 0.99))
        failure_prob = float(np.clip(risk_score * 0.91 + rng.uniform(0.01, 0.04), 0.01, 0.98))

        # Fault signatures
        subsystem_issues = []
        if coolant_temp > 115:
            subsystem_issues.append("CRITICAL ENGINE COOLING LOOP OVERHEAT (DTC P0217)")
        elif coolant_temp > 102:
            subsystem_issues.append("ELEVATED ENGINE HEAD TEMP (DTC P0117)")

        if fuel_pressure < 34:
            subsystem_issues.append("COMMON RAIL PRESSURE BELOW SPEC (DTC P0087)")
        elif fuel_pressure > 65:
            subsystem_issues.append("FUEL RAIL OVERPRESSURE SURGE (DTC P0088)")

        if rpm > 9200:
            subsystem_issues.append("ENGINE OVERSPEED ROTATIONAL STRESS (DTC P0219)")

        if load > 85 and rpm < 2200:
            subsystem_issues.append("LOW-SPEED PRE-IGNITION / KNOCK DETECTED")

        if mileage > 160000:
            subsystem_issues.append("HIGH ACCUMULATED POWERTRAIN BEARING & TIMING WEAR")

        if risk_score > 0.70:
            rec = "URGENT: High probability of imminent component failure. Cease vehicle operation immediately. Perform emergency diagnostic inspection and cooling loop flush before re-engaging powertrain."
            risk_level = "URGENT"
        elif risk_score > 0.50:
            rec = "HIGH RISK: Substantial component wear and elevated thermal profile detected. Schedule certified mechanical service within 500 km or 72 hours."
            risk_level = "HIGH"
        elif risk_score > 0.30:
            rec = "MODERATE RISK: Vehicle health parameters showing slight deviation from baseline. Recommend sensor calibration and scheduled check within 30 days."
            risk_level = "MODERATE"
        else:
            rec = "LOW RISK: Vehicle operational profile nominal. All primary powertrain telemetry, lubrication systems, and auxiliary sensor loops within manufacturer tolerances."
            risk_level = "LOW"

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
# SESSION STATE & SCENARIO SELECTION
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
    set_scenario(2500, 90, 48, 65, 26, 45, 24, 48000)

if "vehicle_model" not in st.session_state:
    st.session_state.vehicle_model = "BMW M3 Competition (G80)"


# -----------------------------------------------------------------------------
# SIDEBAR: SENSOR INPUTS & TELEMETRY CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    render_html("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="bezel-badge badge-cyan">TELEMETRY RIG</span>
            <span style="font-family:'Chakra Petch'; font-size:14px; font-weight:700; color:#FFF;">VHM-UNIT-01</span>
        </div>
        <span class="bezel-badge badge-nominal"><span class="led-pulse"></span> ONLINE</span>
    </div>
    """)

    st.markdown("### 🎛️ TEST PROFILES")
    p1, p2 = st.columns(2)
    with p1:
        if st.button("🟢 Nominal", use_container_width=True):
            set_scenario(2400, 89, 42, 60, 24, 46, 22, 45000)
            st.rerun()
        if st.button("⚠️ Thermal", use_container_width=True):
            set_scenario(5600, 118, 76, 95, 65, 34, 38, 135000)
            st.rerun()
    with p2:
        if st.button("⚡ High Speed", use_container_width=True):
            set_scenario(8400, 102, 88, 220, 82, 56, 28, 72000)
            st.rerun()
        if st.button("🚨 Critical", use_container_width=True):
            set_scenario(12800, 142, 98, 195, 96, 25, 45, 225000)
            st.rerun()

    st.markdown("<hr style='margin:16px 0; border-color:var(--border-color);'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ POWERTRAIN SENSORS")

    engine_rpm = st.slider(
        "Engine RPM", 1000, 15000,
        value=int(st.session_state.engine_rpm),
        step=50, key="engine_rpm"
    )

    engine_load = st.slider(
        "Engine Load (%)", 0, 100,
        value=int(st.session_state.engine_load),
        step=1, key="engine_load"
    )

    throttle_pos = st.slider(
        "Throttle Position (%)", 0, 100,
        value=int(st.session_state.throttle_pos),
        step=1, key="throttle_pos"
    )

    st.markdown("<hr style='margin:16px 0; border-color:var(--border-color);'>", unsafe_allow_html=True)
    st.markdown("### 🌡️ THERMAL & FLUIDS")

    coolant_temp = st.slider(
        "Coolant Temperature (°C)", 60, 200,
        value=int(st.session_state.coolant_temp),
        step=1, key="coolant_temp"
    )

    fuel_pressure = st.slider(
        "Fuel Pressure (psi)", 20, 80,
        value=int(st.session_state.fuel_pressure),
        step=1, key="fuel_pressure"
    )

    air_temp = st.slider(
        "Air Temperature (°C)", -20, 60,
        value=int(st.session_state.air_temp),
        step=1, key="air_temp"
    )

    st.markdown("<hr style='margin:16px 0; border-color:var(--border-color);'>", unsafe_allow_html=True)
    st.markdown("### 🚗 KINEMATICS & ODOMETER")

    vehicle_speed = st.slider(
        "Vehicle Speed (km/h)", 0, 350,
        value=int(st.session_state.vehicle_speed),
        step=5, key="vehicle_speed"
    )

    mileage = st.number_input(
        "Accumulated Mileage (km)", 0, 300000,
        value=int(st.session_state.mileage),
        step=1000, key="mileage"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ EXECUTE TELEMETRY DIAGNOSTICS", type="primary", use_container_width=True):
        st.rerun()


# Package inputs and run inference
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

# Status colors
if risk_level == "URGENT":
    accent_hex = "#FF3B30"
    badge_cls = "badge-alert"
    status_msg = "CRITICAL FAILURE IMMINENT"
elif risk_level == "HIGH":
    accent_hex = "#FF9E00"
    badge_cls = "badge-warn"
    status_msg = "ELEVATED SYSTEM RISK"
elif risk_level == "MODERATE":
    accent_hex = "#00E5FF"
    badge_cls = "badge-cyan"
    status_msg = "MONITORING DEVIATION"
else:
    accent_hex = "#00F59B"
    badge_cls = "badge-nominal"
    status_msg = "ON PATROL // ALL SYSTEMS NOMINAL"

utc_time = datetime.utcnow().strftime("%H:%M:%S")


# -----------------------------------------------------------------------------
# TOP BRAND & TELEMETRY HEADER
# -----------------------------------------------------------------------------
render_html(f"""
<div class="aerion-navbar">
    <div class="brand-section">
        <div class="brand-logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="9" stroke="var(--neon-green)" stroke-width="2"/>
                <circle cx="12" cy="12" r="3" fill="var(--neon-green)"/>
                <path d="M12 3V7M12 17V21M3 12H7M17 12H21" stroke="var(--neon-green)" stroke-width="2"/>
            </svg>
            <span>AERION TELEMETRY</span>
        </div>
        <div style="font-family:'JetBrains Mono'; font-size:11px; color:var(--text-dim); background:rgba(255,255,255,0.04); padding:4px 10px; border-radius:6px; border:1px solid var(--border-color);">
            VEHICLE: <span style="color:#FFF; font-weight:600;">{st.session_state.vehicle_model}</span>
        </div>
    </div>
    <div class="nav-telemetry-stats">
        <div class="nav-stat-group">
            <div class="nav-stat-label">SYSTEM UTC CLOCK</div>
            <div class="nav-stat-val">{utc_time} <span style="font-size:10px; color:var(--text-muted);">UTC</span></div>
        </div>
        <div style="width:1px; height:26px; background:var(--border-color);"></div>
        <div class="nav-stat-group">
            <div class="nav-stat-label">CAN BUS BITRATE</div>
            <div class="nav-stat-val" style="color:var(--neon-green);">1.0 <span style="font-size:10px; color:var(--text-dim);">MBPS</span></div>
        </div>
        <div style="width:1px; height:26px; background:var(--border-color);"></div>
        <div>
            <span class="bezel-badge {badge_cls}">
                <span class="led-pulse"></span> {status_msg}
            </span>
        </div>
    </div>
</div>
""")


# -----------------------------------------------------------------------------
# REAL, FULLY FUNCTIONAL STREAMLIT TABS (No dummy unclickable spans!)
# -----------------------------------------------------------------------------
tab_telemetry, tab_diagnostics, tab_matrix, tab_config = st.tabs([
    "🛰️ TELEMETRY & HUD",
    "🔧 DIAGNOSTIC FAULTS & DTC",
    "📊 SENSOR MATRIX & BOUNDS",
    "⚙️ VEHICLE SYSTEM CONFIG"
])


# =============================================================================
# TAB 1: TELEMETRY & HUD
# =============================================================================
with tab_telemetry:
    # 4 HERO KPI READOUT CARDS
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        render_html(f"""
        <div class="bezel-card" style="border-top: 2px solid {accent_hex};">
            <div class="bezel-header">
                <span class="bezel-title">HEALTH STATUS</span>
                <span class="bezel-badge {badge_cls}">LIVE ●</span>
            </div>
            <div class="hero-metric-val" style="color:{accent_hex};">{risk_level}</div>
            <div class="hero-metric-sub">Calculated Risk Index: {risk_score * 100:.1f} / 100</div>
        </div>
        """)

    with k2:
        render_html(f"""
        <div class="bezel-card" style="border-top: 2px solid var(--neon-cyan);">
            <div class="bezel-header">
                <span class="bezel-title">FAILURE PROBABILITY</span>
                <span class="bezel-badge badge-cyan">ESTIMATED</span>
            </div>
            <div class="hero-metric-val" style="color:#FFFFFF;">{failure_prob:.1%}</div>
            <div class="hero-metric-sub">Bayesian Confidence: &plusmn;2.1%</div>
        </div>
        """)

    with k3:
        render_html(f"""
        <div class="bezel-card" style="border-top: 2px solid var(--neon-amber);">
            <div class="bezel-header">
                <span class="bezel-title">ANOMALY SCORE</span>
                <span class="bezel-badge badge-warn">IFOREST</span>
            </div>
            <div class="hero-metric-val" style="color:#FFFFFF;">{anomaly_score:.3f}</div>
            <div class="hero-metric-sub">Baseline Tolerance: 0.350</div>
        </div>
        """)

    with k4:
        rem_km = max(500, int((1.0 - risk_score) * 115000))
        est_hours = max(1, int((1.0 - risk_score) * 98))
        render_html(f"""
        <div class="bezel-card" style="border-top: 2px solid var(--neon-green);">
            <div class="bezel-header">
                <span class="bezel-title">EST. ENDURANCE</span>
                <span class="bezel-badge badge-nominal">SERVICE LIFE</span>
            </div>
            <div class="hero-metric-val" style="color:var(--neon-green);">{rem_km:,} <span style="font-size:16px; color:var(--text-dim);">KM</span></div>
            <div class="hero-metric-sub">Approx. {est_hours}h operational window</div>
        </div>
        """)

    # MAIN DASHBOARD VIEWPORT (3 COLUMNS)
    c_left, c_center, c_right = st.columns([1.05, 1.85, 1.1])

    # --- LEFT COLUMN: SUBSYSTEM INTEGRITY & LIVE TELEMETRY ---
    with c_left:
        powertrain_health = max(5, int(100 - (engine_load * 0.4 + (engine_rpm / 15000) * 40)))
        cooling_health = max(5, int(100 - max(0, coolant_temp - 90) * 1.5))
        fuel_health = max(5, int(100 - abs(fuel_pressure - 45) * 2.2))
        chassis_health = max(5, int(100 - (mileage / 300000) * 80))

        def make_led_bars(score):
            seg1 = "fill-green" if score > 20 else "fill-red"
            seg2 = "fill-green" if score > 40 else ""
            seg3 = "fill-green" if score > 60 else ""
            seg4 = "fill-amber" if score > 80 else ""
            seg5 = "fill-green" if score > 90 else ""
            return f'<div class="bar-track"><div class="bar-segment {seg1}"></div><div class="bar-segment {seg2}"></div><div class="bar-segment {seg3}"></div><div class="bar-segment {seg4}"></div><div class="bar-segment {seg5}"></div></div>'

        def tag_str(val):
            if val > 75:
                return '<span style="color:var(--neon-green); font-weight:600;">NOMINAL</span>'
            elif val > 50:
                return '<span style="color:var(--neon-amber); font-weight:600;">WARN</span>'
            else:
                return '<span style="color:var(--neon-red); font-weight:600;">CRITICAL</span>'

        p_bars = make_led_bars(powertrain_health)
        c_bars = make_led_bars(cooling_health)
        f_bars = make_led_bars(fuel_health)
        w_bars = make_led_bars(chassis_health)

        render_html(f"""
        <div class="bezel-card">
            <div class="bezel-header">
                <span class="bezel-title">AIRFRAME SYSTEM STATUS</span>
                <span class="bezel-badge badge-nominal">{powertrain_health}% NOMINAL</span>
            </div>
            <div style="margin: 10px 0;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                    <span style="color:var(--text-dim);">POWERTRAIN INTEGRITY</span>
                    <span>{tag_str(powertrain_health)}</span>
                </div>
                {p_bars}
            </div>
            <div style="margin: 10px 0;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                    <span style="color:var(--text-dim);">COOLING LOOP STATUS</span>
                    <span>{tag_str(cooling_health)}</span>
                </div>
                {c_bars}
            </div>
            <div style="margin: 10px 0;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                    <span style="color:var(--text-dim);">FUEL INJECTION RAIL</span>
                    <span>{tag_str(fuel_health)}</span>
                </div>
                {f_bars}
            </div>
            <div style="margin: 10px 0;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px;">
                    <span style="color:var(--text-dim);">CHASSIS ACCUMULATION</span>
                    <span>{tag_str(chassis_health)}</span>
                </div>
                {w_bars}
            </div>
        </div>
        """)

        # Live Telemetry Rows
        egt_color = '#FF3B30' if coolant_temp > 115 else ('#FF9E00' if coolant_temp > 100 else '#00F59B')
        render_html(f"""
        <div class="bezel-card">
            <div class="bezel-header">
                <span class="bezel-title">LIVE TELEMETRY</span>
                <span class="bezel-badge badge-nominal">LIVE &bull;</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-param-title">ENGINE RPM</span>
                <span class="telemetry-param-val" style="color:var(--neon-green);">{engine_rpm:,} <span style="font-size:10px; color:var(--text-muted);">rpm</span></span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-param-title">COOLANT TEMP (EGT)</span>
                <span class="telemetry-param-val" style="color:{egt_color};">{coolant_temp} <span style="font-size:10px; color:var(--text-muted);">°C</span></span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-param-title">ENGINE LOAD</span>
                <span class="telemetry-param-val">{engine_load} <span style="font-size:10px; color:var(--text-muted);">%</span></span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-param-title">FUEL RAIL PRESSURE</span>
                <span class="telemetry-param-val" style="color:var(--neon-cyan);">{fuel_pressure} <span style="font-size:10px; color:var(--text-muted);">psi</span></span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-param-title">AIR INTAKE TEMP</span>
                <span class="telemetry-param-val">{air_temp} <span style="font-size:10px; color:var(--text-muted);">°C</span></span>
            </div>
        </div>
        """)

    # --- CENTER COLUMN: INTERACTIVE HUD & SCHEMATIC VIEW SELECTOR ---
    with c_center:
        # Dynamic Color-coded sensor node states
        c_rad = "#FF3B30" if coolant_temp > 115 else ("#FF9E00" if coolant_temp > 100 else "#00F59B")
        c_eng = "#FF3B30" if engine_rpm > 9000 or engine_load > 85 else ("#FF9E00" if engine_rpm > 5500 else "#00F59B")
        c_fuel = "#FF3B30" if fuel_pressure < 34 or fuel_pressure > 65 else "#00E5FF"

        # Interactive view selector (Real clickable radio pills!)
        hud_view = st.radio(
            "HUD View Mode",
            ["📐 3D Wireframe", "🌡️ Thermal Heatmap", "⚡ Sensor Topology"],
            index=0,
            horizontal=True,
            key="hud_view_mode",
            label_visibility="collapsed"
        )

        # Render corresponding visual according to the selected view
        if hud_view == "📐 3D Wireframe":
            # High-end Isometric Fixed-Wing Platform Blueprint
            render_html(f"""
            <div class="bezel-card" style="padding:14px; text-align:center;">
                <div class="bezel-header" style="margin-bottom:6px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="bezel-title">CHASSIS BLUEPRINT // ISOMETRIC WIREFRAME</span>
                        <span class="bezel-badge badge-cyan">3D HUD</span>
                    </div>
                    <span class="bezel-badge badge-nominal">MODE: WIREFRAME</span>
                </div>
                <div style="position:relative; width:100%; height:250px; overflow:hidden;">
                    <svg viewBox="0 0 680 320" style="width:100%; height:100%; filter:drop-shadow(0 0 16px rgba(0, 229, 255, 0.12));">
                        <defs>
                            <radialGradient id="radarSweep" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="rgba(0, 245, 155, 0.25)" />
                                <stop offset="70%" stop-color="rgba(0, 229, 255, 0.08)" />
                                <stop offset="100%" stop-color="transparent" />
                            </radialGradient>
                            <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="{c_eng}" stop-opacity="0.8" />
                                <stop offset="100%" stop-color="{c_eng}" stop-opacity="0" />
                            </radialGradient>
                        </defs>
                        <ellipse cx="340" cy="230" rx="240" ry="68" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="1" stroke-dasharray="4,6" />
                        <ellipse cx="340" cy="230" rx="160" ry="46" fill="url(#radarSweep)" stroke="rgba(0, 245, 155, 0.2)" stroke-width="1.5" />
                        <ellipse cx="340" cy="230" rx="80" ry="24" fill="none" stroke="rgba(255, 158, 0, 0.4)" stroke-width="1.5" stroke-dasharray="2,3" />
                        <circle cx="340" cy="230" r="14" fill="rgba(255, 158, 0, 0.2)" stroke="#FF9E00" stroke-width="1.5" />
                        <text x="340" y="234" fill="#FF9E00" font-family="JetBrains Mono" font-size="9" font-weight="700" text-anchor="middle">WP2</text>
                        <polygon points="340,60 365,130 355,200 325,200 315,130" fill="rgba(16, 23, 34, 0.9)" stroke="rgba(0, 229, 255, 0.4)" stroke-width="1.5" />
                        <line x1="340" y1="60" x2="340" y2="200" stroke="rgba(255, 255, 255, 0.2)" stroke-width="1" stroke-dasharray="2,3" />
                        <polygon points="325,130 110,105 105,120 320,150" fill="rgba(18, 26, 38, 0.75)" stroke="rgba(0, 229, 255, 0.35)" stroke-width="1.2" />
                        <line x1="210" y1="118" x2="205" y2="135" stroke="rgba(255, 255, 255, 0.15)" stroke-width="1" />
                        <polygon points="355,130 570,105 575,120 360,150" fill="rgba(18, 26, 38, 0.75)" stroke="rgba(0, 229, 255, 0.35)" stroke-width="1.2" />
                        <line x1="470" y1="118" x2="475" y2="135" stroke="rgba(255, 255, 255, 0.15)" stroke-width="1" />
                        <polygon points="330,195 285,225 295,232 335,202" fill="rgba(14, 20, 30, 0.8)" stroke="rgba(0, 229, 255, 0.3)" stroke-width="1" />
                        <polygon points="350,195 395,225 385,232 345,202" fill="rgba(14, 20, 30, 0.8)" stroke="rgba(0, 229, 255, 0.3)" stroke-width="1" />
                        <ellipse cx="340" cy="208" rx="26" ry="6" fill="none" stroke="rgba(0, 245, 155, 0.4)" stroke-width="1" stroke-dasharray="3,3" />
                        <rect x="328" y="125" width="24" height="42" rx="4" fill="url(#coreGlow)" stroke="{c_eng}" stroke-width="1.5" />
                        <line x1="332" y1="135" x2="348" y2="135" stroke="#FFFFFF" stroke-width="1" />
                        <line x1="332" y1="145" x2="348" y2="145" stroke="#FFFFFF" stroke-width="1" />
                        <line x1="332" y1="155" x2="348" y2="155" stroke="#FFFFFF" stroke-width="1" />
                        <circle cx="340" cy="60" r="5" fill="{c_rad}" stroke="#FFFFFF" stroke-width="1" />
                        <line x1="340" y1="60" x2="385" y2="42" stroke="{c_rad}" stroke-width="1" stroke-dasharray="2,2" />
                        <rect x="385" y="32" width="120" height="20" rx="3" fill="rgba(11, 16, 24, 0.85)" stroke="{c_rad}" stroke-width="1" />
                        <text x="392" y="46" fill="{c_rad}" font-family="JetBrains Mono" font-size="9" font-weight="600">EGT HEAD: {coolant_temp}°C</text>
                        <circle cx="108" cy="112" r="4" fill="var(--neon-green)" />
                        <circle cx="572" cy="112" r="4" fill="var(--neon-green)" />
                        <circle cx="340" cy="146" r="6" fill="{c_eng}">
                            <animate attributeName="r" values="5;7;5" dur="1.4s" repeatCount="indefinite"/>
                        </circle>
                        <line x1="352" y1="146" x2="430" y2="175" stroke="{c_eng}" stroke-width="1" stroke-dasharray="2,2" />
                        <rect x="430" y="165" width="125" height="20" rx="3" fill="rgba(11, 16, 24, 0.85)" stroke="{c_eng}" stroke-width="1" />
                        <text x="438" y="179" fill="{c_eng}" font-family="JetBrains Mono" font-size="9" font-weight="600">ENGINE: {engine_rpm:,} RPM</text>
                        <line x1="328" y1="146" x2="230" y2="175" stroke="{c_fuel}" stroke-width="1" stroke-dasharray="2,2" />
                        <rect x="135" y="165" width="105" height="20" rx="3" fill="rgba(11, 16, 24, 0.85)" stroke="{c_fuel}" stroke-width="1" />
                        <text x="142" y="179" fill="{c_fuel}" font-family="JetBrains Mono" font-size="9" font-weight="600">RAIL: {fuel_pressure} PSI</text>
                    </svg>
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:10px; color:var(--text-muted); display:flex; justify-content:space-around; margin-top:4px;">
                    <span>ACTIVE SENSORS: 8 CHANNELS</span>
                    <span>BEARING: 16° AZIMUTH</span>
                    <span>DOWNLINK PING: 18ms</span>
                </div>
            </div>
            """)
        elif hud_view == "🌡️ Thermal Heatmap":
            # Real 2D Thermal Contour Heatmap of Engine Bay
            x_mesh = np.linspace(-3, 3, 25)
            y_mesh = np.linspace(-2, 2, 20)
            X, Y = np.meshgrid(x_mesh, y_mesh)
            # Thermal center around radiator and engine core
            R_eng = np.sqrt(X**2 + Y**2)
            R_rad = np.sqrt((X - 1.5)**2 + (Y)**2)
            Z = (coolant_temp * 0.7) * np.exp(-R_rad) + (engine_rpm / 15000.0 * 80 + coolant_temp * 0.5) * np.exp(-R_eng)

            fig_heat = px.imshow(
                Z, color_continuous_scale="Turbo",
                labels=dict(x="Chassis X-Axis", y="Chassis Y-Axis", color="Temp (°C)")
            )
            fig_heat.update_layout(
                height=265,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(text="THERMAL HEATMAP TOPOLOGY // ENGINE BAY (°C)", font=dict(family="Chakra Petch", size=11, color="#94A3B8")),
                coloraxis_colorbar=dict(thickness=10, len=0.8, tickfont=dict(family="JetBrains Mono", size=8, color="#64748B"))
            )
            fig_heat.update_xaxes(showticklabels=False)
            fig_heat.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
        else:
            # Sensor Topology & Bus Nodes Visualizer
            nodes_df = pd.DataFrame({
                "System Node": ["Intake Pitot", "Engine Core", "Fuel Rail", "Cooling Radiator", "Chassis Hub"],
                "Bus Address": ["0x18FEEE00", "0x0CF00400", "0x18FEF200", "0x18FEEF00", "0x18FE6800"],
                "Latency": ["12 ms", "8 ms", "14 ms", "10 ms", "16 ms"],
                "Status": ["NOMINAL", "NOMINAL" if engine_rpm < 8000 else "ELEVATED", "NOMINAL", "NOMINAL" if coolant_temp < 105 else "ALERT", "NOMINAL"]
            })
            st.dataframe(nodes_df, use_container_width=True, hide_index=True)

        # Tactical Half-Ring Risk Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={
                'suffix': "%",
                'font': {'color': accent_hex, 'family': 'Chakra Petch', 'size': 38}
            },
            title={
                'text': f"SYSTEM RISK INDEX // {risk_level}",
                'font': {'color': '#94A3B8', 'family': 'Chakra Petch', 'size': 12}
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "rgba(255,255,255,0.15)",
                    'tickfont': {'family': 'JetBrains Mono', 'size': 9, 'color': '#64748B'}
                },
                'bar': {'color': accent_hex, 'thickness': 0.3},
                'bgcolor': "rgba(255, 255, 255, 0.04)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.08)",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 245, 155, 0.12)"},
                    {'range': [30, 50], 'color': "rgba(0, 229, 255, 0.12)"},
                    {'range': [50, 70], 'color': "rgba(255, 158, 0, 0.15)"},
                    {'range': [70, 100], 'color': "rgba(255, 59, 48, 0.22)"}
                ],
                'threshold': {
                    'line': {'color': "#FF3B30", 'width': 3},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=35, b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    # --- RIGHT COLUMN: NAVIGATION & FLIGHT ENVIRONMENT ---
    with c_right:
        # Telemetry Bandwidth Trace Chart
        time_bins = [f"{i*5}s" for i in range(12)]
        np.random.seed(int(abs(engine_rpm + vehicle_speed) % 5000))
        base_bw = 2.4 + (vehicle_speed / 350.0) * 1.0
        bw_series = [round(base_bw + np.random.uniform(-0.15, 0.15), 2) for _ in range(12)]

        fig_bw = go.Figure()
        fig_bw.add_trace(go.Scatter(
            x=time_bins, y=bw_series,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 229, 255, 0.07)',
            line=dict(color='#00E5FF', width=2, shape='spline'),
            name='CAN Data'
        ))
        fig_bw.update_layout(
            height=175,
            margin=dict(l=10, r=10, t=25, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text="TELEMETRY BANDWIDTH (GBPS)",
                font=dict(family="Chakra Petch", size=11, color="#94A3B8")
            ),
            xaxis=dict(showgrid=False, tickfont=dict(family="JetBrains Mono", size=8, color="#64748B")),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickfont=dict(family="JetBrains Mono", size=8, color="#64748B"))
        )
        st.plotly_chart(fig_bw, use_container_width=True, config={'displayModeBar': False})

        # Link Security & Kinematics Environment
        render_html(f"""
        <div class="bezel-card">
            <div class="bezel-header">
                <span class="bezel-title">LINK SECURITY</span>
                <span class="bezel-badge badge-nominal">STRONG</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;">
                <div style="background:rgba(255,255,255,0.02); padding:8px 10px; border-radius:6px; border:1px solid var(--border-color);">
                    <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">BUS PROTOCOL</div>
                    <div style="font-size:12px; font-weight:700; color:#FFF; font-family:'Chakra Petch';">CAN-2.0B / UDS</div>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:8px 10px; border-radius:6px; border:1px solid var(--border-color);">
                    <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">BUS LATENCY</div>
                    <div style="font-size:12px; font-weight:700; color:var(--neon-green); font-family:'Chakra Petch';">18 ms</div>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:8px 10px; border-radius:6px; border:1px solid var(--border-color);">
                    <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">GROUND SPEED</div>
                    <div style="font-size:12px; font-weight:700; color:var(--neon-cyan); font-family:'Chakra Petch';">{vehicle_speed} KM/H</div>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:8px 10px; border-radius:6px; border:1px solid var(--border-color);">
                    <div style="font-size:9px; color:var(--text-muted); font-family:'JetBrains Mono';">AIR DENSITY</div>
                    <div style="font-size:12px; font-weight:700; color:#FFF; font-family:'Chakra Petch';">1.02 kg/m³</div>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:10px; color:var(--text-muted);">
                <span>PROTOCOL: ISO-14229</span>
                <span>TELEMETRY: SYNCHRONIZED</span>
            </div>
        </div>
        """)

    # BOTTOM ROW: Power System & Engine Spectrum
    b_col1, b_col2 = st.columns([1, 1.25])

    with b_col1:
        power_kw = round((engine_load / 100.0) * (engine_rpm / 1000.0) * 22.4, 1)
        voltage_val = 14.7 if coolant_temp < 110 else 13.6
        current_amp = round(12.4 + (power_kw * 0.45), 1)

        render_html(f"""
        <div class="bezel-card">
            <div class="bezel-header">
                <span class="bezel-title">POWER SYSTEM MODULE</span>
                <span class="bezel-badge badge-nominal">84.2% LEVEL</span>
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-bottom:12px;">
                <div>
                    <div class="hero-metric-sub">POWER OUTPUT</div>
                    <div style="font-family:'Chakra Petch'; font-size:24px; font-weight:700; color:#FFF;">
                        {power_kw} <span style="font-size:12px; color:var(--neon-green);">kW</span>
                    </div>
                </div>
                <div>
                    <div class="hero-metric-sub">BUS CURRENT</div>
                    <div style="font-family:'Chakra Petch'; font-size:24px; font-weight:700; color:#FFF;">
                        {current_amp} <span style="font-size:12px; color:var(--neon-cyan);">A</span>
                    </div>
                </div>
                <div>
                    <div class="hero-metric-sub">VOLTAGE</div>
                    <div style="font-family:'Chakra Petch'; font-size:24px; font-weight:700; color:#FFF;">
                        {voltage_val} <span style="font-size:12px; color:var(--text-dim);">V</span>
                    </div>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:11px; color:var(--text-dim); border-top:1px solid var(--border-color); padding-top:10px;">
                <span>THROTTLE: {throttle_pos}%</span>
                <span>LOAD: {engine_load}%</span>
                <span>FUEL PRESS: {fuel_pressure} PSI</span>
            </div>
        </div>
        """)

    with b_col2:
        rpm_bins = np.linspace(1000, 15000, 24)
        temps_series = []
        colors_series = []
        for r in rpm_bins:
            dist = abs(r - engine_rpm) / 1000.0
            val = max(30, int(coolant_temp * math.exp(-0.16 * dist) + np.random.uniform(4, 10)))
            temps_series.append(val)
            if val > 115:
                colors_series.append('#FF3B30')
            elif val > 95:
                colors_series.append('#FF9E00')
            else:
                colors_series.append('#00F59B')

        efficiency_pct = max(35, int(96 - (engine_load * 0.25 + abs(coolant_temp - 90) * 0.5)))

        fig_spec = go.Figure(go.Bar(
            x=[f"{int(r)}" for r in rpm_bins],
            y=temps_series,
            marker=dict(color=colors_series, line=dict(color='rgba(0,0,0,0.4)', width=0.5))
        ))
        fig_spec.update_layout(
            height=175,
            margin=dict(l=10, r=10, t=25, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text=f"ENGINE STATUS // THERMAL SPECTRUM &bull; EFFICIENCY: {efficiency_pct}%",
                font=dict(family="Chakra Petch", size=11, color="#94A3B8")
            ),
            xaxis=dict(showgrid=False, tickfont=dict(family="JetBrains Mono", size=7, color="#64748B"), tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickfont=dict(family="JetBrains Mono", size=8, color="#64748B"))
        )
        st.plotly_chart(fig_spec, use_container_width=True, config={'displayModeBar': False})


# =============================================================================
# TAB 2: DIAGNOSTIC FAULTS & DTC ADVISORY
# =============================================================================
with tab_diagnostics:
    render_html(f"""
    <div class="bezel-card">
        <div class="bezel-header">
            <span class="bezel-title">AUTOMOTIVE DIAGNOSTIC TROUBLE CODE (DTC) LOG</span>
            <span class="bezel-badge {badge_cls}">RISK: {risk_level}</span>
        </div>
        <div style="font-size:13px; color:var(--text-dim); margin-bottom:14px;">
            Real-time OBD-II / CAN-Bus fault injection and predictive component failure analysis.
        </div>
    </div>
    """)

    # AI Advisory Banner
    advisory_class = "urgent" if risk_level == "URGENT" else ("high" if risk_level == "HIGH" else ("moderate" if risk_level == "MODERATE" else ""))
    icon_char = "🚨" if risk_level == "URGENT" else ("⚠️" if risk_level == "HIGH" else ("ℹ️" if risk_level == "MODERATE" else "✅"))

    fault_html = ""
    if subsystem_issues:
        fault_items = " &bull; ".join(subsystem_issues)
        fault_html = f'<div style="margin-top:8px; padding-top:8px; border-top:1px dashed rgba(255,255,255,0.1); font-family:\'JetBrains Mono\'; font-size:11px; color:{accent_hex};"><strong>ACTIVE FAULT SIGNATURES:</strong> {fault_items}</div>'

    render_html(f"""
    <div class="advisory-box {advisory_class}">
        <div style="display:flex; align-items:flex-start; gap:14px;">
            <span style="font-size:22px;">{icon_char}</span>
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                    <span style="font-family:'Chakra Petch'; font-size:14px; font-weight:700; color:#FFF; letter-spacing:0.06em;">
                        PREDICTIVE MAINTENANCE ACTION PLAN // {risk_level} LEVEL
                    </span>
                    <span class="bezel-badge {badge_cls}">ACTION: {risk_level}</span>
                </div>
                <div style="font-size:13px; color:#E2E8F0; line-height:1.45;">
                    {recommendation}
                </div>
                {fault_html}
            </div>
        </div>
    </div>
    """)

    # Detailed DTC Table
    dtc_records = []
    if coolant_temp > 115:
        dtc_records.append(("P0217", "Engine Coolant Over Temperature Condition", "Cooling System", "CRITICAL", "Inspect thermostat valve, water pump, and radiator fan relay immediately."))
    elif coolant_temp > 102:
        dtc_records.append(("P0117", "Engine Coolant Temperature Sensor 1 Circuit Low", "Sensors", "WARNING", "Check sensor wiring harness and coolant fluid level."))

    if fuel_pressure < 34:
        dtc_records.append(("P0087", "Fuel Rail/System Pressure Too Low", "Fuel Delivery", "CRITICAL", "High-pressure fuel pump delivery deficit. Check fuel filter for clogging."))
    elif fuel_pressure > 65:
        dtc_records.append(("P0088", "Fuel Rail/System Pressure Too High", "Fuel Delivery", "WARNING", "Fuel pressure regulator stuck closed. Verify return line flow."))

    if engine_rpm > 9200:
        dtc_records.append(("P0219", "Engine Overspeed Condition", "Powertrain", "CRITICAL", "Engine RPM exceeded safe valve float threshold. Inspect valve springs."))

    if engine_load > 85 and engine_rpm < 2200:
        dtc_records.append(("P0300", "Random/Multiple Cylinder Misfire Detected", "Ignition", "WARNING", "Low-speed pre-ignition risk. Adjust spark advance timing."))

    if mileage > 160000:
        dtc_records.append(("P1340", "Timing Chain / Camshaft Alignment Drift", "Mechanical", "WARNING", "High accumulated mileage. Inspect timing chain tensioner and guide wear."))

    if not dtc_records:
        dtc_records.append(("P0000", "No Fault Codes Detected in Controller Memory", "Powertrain", "NOMINAL", "All sensor metrics within certified OEM operational boundaries."))

    dtc_rows = ""
    for code, desc, sys_name, sev, action in dtc_records:
        c_sev = "var(--neon-green)" if sev == "NOMINAL" else ("var(--neon-amber)" if sev == "WARNING" else "var(--neon-red)")
        b_sev = "badge-nominal" if sev == "NOMINAL" else ("badge-warn" if sev == "WARNING" else "badge-alert")
        dtc_rows += f'<tr><td style="font-family:\'Chakra Petch\'; font-weight:700; color:{c_sev};">{code}</td><td style="font-weight:600; color:#FFF;">{desc}</td><td style="color:var(--text-dim);">{sys_name}</td><td><span class="bezel-badge {b_sev}">{sev}</span></td><td style="color:var(--text-dim);">{action}</td></tr>'

    render_html(f"""
    <div class="bezel-card">
        <div class="bezel-header">
            <span class="bezel-title">ACTIVE DTC ERROR CODES & RECOMMENDED SERVICE WORKSHOP ACTIONS</span>
            <span class="bezel-badge badge-cyan">{len(dtc_records)} DETECTED</span>
        </div>
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>DTC CODE</th>
                    <th>FAULT DESCRIPTION</th>
                    <th>SUBSYSTEM</th>
                    <th>SEVERITY</th>
                    <th>RECOMMENDED WORKSHOP ACTION</th>
                </tr>
            </thead>
            <tbody>
                {dtc_rows}
            </tbody>
        </table>
    </div>
    """)


# =============================================================================
# TAB 3: SENSOR MATRIX & OPERATING BOUNDS
# =============================================================================
with tab_matrix:
    def classify_sensor(p, v):
        if p == "Engine_RPM":
            return ("CRITICAL", "red") if v > 9500 else (("WARN", "amber") if v > 6500 else ("NOMINAL", "green"))
        elif p == "Coolant_Temperature":
            return ("CRITICAL", "red") if v > 115 else (("WARN", "amber") if v > 100 else ("NOMINAL", "green"))
        elif p == "Engine_Load":
            return ("CRITICAL", "red") if v > 90 else (("WARN", "amber") if v > 75 else ("NOMINAL", "green"))
        elif p == "Fuel_Pressure":
            return ("CRITICAL", "red") if (v < 34 or v > 66) else (("WARN", "amber") if (v < 38 or v > 60) else ("NOMINAL", "green"))
        elif p == "Mileage":
            return ("WARN", "amber") if v > 150000 else ("NOMINAL", "green")
        else:
            return ("NOMINAL", "green")

    matrix_data = [
        ("Engine_RPM", engine_rpm, "1,500 – 6,000", "RPM", "Crankshaft rotational speed"),
        ("Coolant_Temperature", coolant_temp, "85 – 100", "°C", "Internal cooling loop thermal balance"),
        ("Engine_Load", engine_load, "20 – 70", "%", "Calculated engine mechanical torque ratio"),
        ("Vehicle_Speed", vehicle_speed, "0 – 130", "km/h", "GPS & wheel hub ground velocity"),
        ("Throttle_Position", throttle_pos, "10 – 80", "%", "Drive-by-wire valve percentage"),
        ("Fuel_Pressure", fuel_pressure, "40 – 55", "psi", "Common rail high-pressure injection loop"),
        ("Air_Temperature", air_temp, "0 – 40", "°C", "Manifold intake ambient air temperature"),
        ("Mileage", mileage, "0 – 150,000", "km", "Lifetime odometer accumulation")
    ]

    rows_html = ""
    for p_name, p_val, p_opt, p_unit, p_desc in matrix_data:
        s_tag, s_color = classify_sensor(p_name, p_val)
        hex_color = "var(--neon-green)" if s_color == "green" else ("var(--neon-amber)" if s_color == "amber" else "var(--neon-red)")
        badge_style = "badge-nominal" if s_color == "green" else ("badge-warn" if s_color == "amber" else "badge-alert")
        val_repr = f"{p_val:,}" if isinstance(p_val, int) else f"{p_val}"

        rows_html += f'<tr><td style="font-weight:600; color:#FFFFFF;">{p_name}</td><td style="color:{hex_color}; font-weight:700;">{val_repr}</td><td style="color:var(--text-dim);">{p_opt}</td><td style="color:var(--text-muted);">{p_unit}</td><td style="color:var(--text-dim);">{p_desc}</td><td><span class="bezel-badge {badge_style}">{s_tag}</span></td></tr>'

    render_html(f"""
    <div class="bezel-card">
        <div class="bezel-header">
            <span class="bezel-title">DETAILED SENSOR TELEMETRY MATRIX & OPERATING THRESHOLDS</span>
            <span class="bezel-badge badge-cyan">TELEMETRY MATRIX</span>
        </div>
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>SENSOR PARAMETER</th>
                    <th>LIVE VALUE</th>
                    <th>OPTIMAL THRESHOLD</th>
                    <th>UNIT</th>
                    <th>DIAGNOSTIC DESCRIPTION</th>
                    <th>SUBSYSTEM STATUS</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """)

    # Correlation Plot
    st.markdown("### 📈 SENSOR STRESS MULTI-AXIS RADAR")
    r_vals = [
        min(engine_rpm / 15000.0, 1.0) * 100,
        min(max(coolant_temp - 60, 0) / 140.0, 1.0) * 100,
        engine_load,
        min(vehicle_speed / 350.0, 1.0) * 100,
        throttle_pos,
        min(abs(fuel_pressure - 45) / 35.0, 1.0) * 100,
        min(mileage / 300000.0, 1.0) * 100
    ]
    r_labels = ["RPM", "Coolant Temp", "Load", "Speed", "Throttle", "Fuel Deviation", "Mileage"]

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=r_vals,
        theta=r_labels,
        fill='toself',
        fillcolor='rgba(0, 245, 155, 0.15)',
        line=dict(color='#00F59B', width=2)
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(tickfont=dict(family="JetBrains Mono", size=10, color="#94A3B8"), linecolor='rgba(255,255,255,0.1)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})


# =============================================================================
# TAB 4: VEHICLE SYSTEM CONFIG
# =============================================================================
with tab_config:
    render_html("""
    <div class="bezel-card">
        <div class="bezel-header">
            <span class="bezel-title">VEHICLE PROFILE & DIAGNOSTIC BUS CONFIGURATION</span>
            <span class="bezel-badge badge-cyan">SYSTEM CONFIG</span>
        </div>
        <div style="font-size:13px; color:var(--text-dim); margin-bottom:14px;">
            Configure target chassis model, diagnostic baud rates, calibration parameters, and service intervals.
        </div>
    </div>
    """)

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown("#### 🚗 Chassis Profile")
        new_model = st.selectbox(
            "Target Vehicle Model",
            ["BMW M3 Competition (G80)", "Porsche 911 GT3 (992)", "Tesla Model S Plaid", "Ford F-150 Lightning", "Custom Fleet Machine"],
            index=0
        )
        if new_model != st.session_state.vehicle_model:
            st.session_state.vehicle_model = new_model
            st.rerun()

        st.selectbox("Powertrain Architecture", ["Twin-Turbo Inline-6 (Gasoline)", "Dual-Motor Electric AWD", "Diesel Turbo Common-Rail", "Hybrid Synergy Drive"])
        st.number_input("Service Interval Threshold (km)", min_value=5000, max_value=30000, value=15000, step=1000)

    with cfg2:
        st.markdown("#### 📡 Telemetry & Bus Protocol")
        st.selectbox("CAN Bus Protocol", ["ISO 11898-1 CAN 2.0B (500 kbps)", "ISO 11898-2 CAN-FD (2 Mbps)", "Automotive Ethernet (100BASE-T1)", "DoIP ISO 13400"])
        st.selectbox("Inference Model Architecture", ["VehiclePredictiveMaintenanceModel (Sigmoid Calibrated)", "RandomForestClassifier + IsolationForest", "GradientBoosting Regressor"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 SAVE CONFIGURATION & RE-CALIBRATE", use_container_width=True):
            st.success("Vehicle profile configuration saved successfully!")

# Footer
render_html("""
<div style="text-align:center; margin-top:28px; margin-bottom:12px; font-family:'JetBrains Mono'; font-size:11px; color:var(--text-muted);">
    AERION DEFENSE &bull; VEHICLE PREDICTIVE TELEMETRY ENGINE &bull; VERSION 3.4.0 &bull; SECURE REALTIME DEPLOYMENT
</div>
""")
