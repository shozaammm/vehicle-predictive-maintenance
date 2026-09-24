import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
from speedx_assets import CAR_B64, RADAR_B64

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpeedX // Vehicle Predictive Telemetry",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render pure HTML without markdown indentation interference
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
# SPEEDX BESPOKE AUTOMOTIVE STYLING (Matching 1bcabb903ce3b1e3dbee4699de22c9d9.webp)
# -----------------------------------------------------------------------------
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* SpeedX Palette */
:root {
    --bg-cockpit: #0D0F13;
    --card-surface: #14171E;
    --card-surface-subtle: #1A1E27;
    --card-border: rgba(255, 255, 255, 0.06);
    --speedx-green: #22C55E;
    --speedx-green-glow: rgba(34, 197, 94, 0.4);
    --speedx-amber: #F97316;
    --speedx-red: #EF4444;
    --text-white: #FFFFFF;
    --text-muted: #8E9BAE;
    --text-dim: #64748B;
}

/* Background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-cockpit) !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(34, 197, 94, 0.03) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(249, 115, 22, 0.02) 0%, transparent 40%) !important;
    color: var(--text-white) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Machined Dark Terminal */
[data-testid="stSidebar"] {
    background-color: #101217 !important;
    border-right: 1px solid var(--card-border) !important;
}

/* SpeedX Master Top Header */
.speedx-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background: var(--card-surface);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.speedx-logo {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    display: flex;
    align-items: center;
}

.speedx-logo-x {
    color: var(--speedx-green);
    margin-left: 1px;
}

.speedx-pills-group {
    display: flex;
    align-items: center;
    gap: 10px;
}

.speedx-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1A1E27;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #CBD5E1;
}

.speedx-icon-btn {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--speedx-green);
    color: #000000;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    box-shadow: 0 0 12px var(--speedx-green-glow);
}

/* SpeedX Card Architecture */
.speedx-card {
    background: var(--card-surface);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 18px;
    position: relative;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.speedx-card-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}

/* Neon Green Progress Bars */
.speedx-bar-container {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
}

.speedx-bar-fill {
    background: linear-gradient(90deg, #16A34A 0%, #22C55E 100%);
    box-shadow: 0 0 10px var(--speedx-green-glow);
    height: 100%;
    border-radius: 6px;
}

/* 5 Vertical Segment Blocks */
.speedx-segments {
    display: flex;
    gap: 5px;
    margin-top: 8px;
}

.speedx-segment-block {
    flex: 1;
    height: 18px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}

.speedx-segment-block.lit {
    background: var(--speedx-green);
    box-shadow: 0 0 8px var(--speedx-green-glow);
}

/* Comms Quote Box */
.speedx-quote-box {
    background: #1A1E27;
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin: 10px 0;
    font-size: 13px;
    line-height: 1.5;
    color: #E2E8F0;
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
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
}

.matrix-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.matrix-table tr:hover td {
    background: rgba(255, 255, 255, 0.02);
}

/* Streamlit Tabs Customization */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #14171E !important;
    padding: 6px !important;
    border-radius: 12px !important;
    gap: 8px !important;
    border: 1px solid var(--card-border) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #94A3B8 !important;
    border: none !important;
    padding: 8px 18px !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background: #22C55E !important;
    color: #000000 !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 12px var(--speedx-green-glow) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"], div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* Sidebar Sliders & Buttons */
div[data-baseweb="slider"] {
    padding: 8px 0 !important;
}

.stButton > button {
    background: #1A1E27 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    padding: 8px 14px !important;
}

.stButton > button:hover {
    border-color: var(--speedx-green) !important;
    background: #202632 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #22C55E 0%, #16A34A 100%) !important;
    color: #000000 !important;
    border-color: var(--speedx-green) !important;
    box-shadow: 0 0 16px var(--speedx-green-glow) !important;
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

        # Fuel pressure anomaly penalty
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

        seed = int(abs(rpm * 3 + coolant_temp * 7 + speed * 11 + mileage) % 99999)
        rng = np.random.RandomState(seed)
        anomaly_score = float(np.clip(0.08 + (risk_score * 0.74) + rng.uniform(-0.02, 0.02), 0.02, 0.99))
        failure_prob = float(np.clip(risk_score * 0.91 + rng.uniform(0.01, 0.04), 0.01, 0.98))

        # Actionable fault signatures
        subsystem_issues = []
        if coolant_temp > 115:
            subsystem_issues.append("CRITICAL ENGINE OVERHEAT (DTC P0217)")
        elif coolant_temp > 102:
            subsystem_issues.append("ELEVATED HEAD TEMPERATURE (DTC P0117)")

        if fuel_pressure < 34:
            subsystem_issues.append("LOW COMMON RAIL PRESSURE (DTC P0087)")
        elif fuel_pressure > 65:
            subsystem_issues.append("FUEL SYSTEM OVERPRESSURE (DTC P0088)")

        if rpm > 9200:
            subsystem_issues.append("ENGINE ROTATIONAL OVERSPEED (DTC P0219)")

        if load > 85 and rpm < 2200:
            subsystem_issues.append("LOW-SPEED PRE-IGNITION DETECTED")

        if mileage > 160000:
            subsystem_issues.append("HIGH TIMING & BEARING WEAR")

        if risk_score > 0.70:
            rec = "URGENT: High probability of component failure. Cease vehicle operation immediately. Inspect cooling loop and fuel rail."
            risk_level = "URGENT"
        elif risk_score > 0.50:
            rec = "HIGH RISK: Substantial component wear detected. Schedule mechanic service check within 500 km."
            risk_level = "HIGH"
        elif risk_score > 0.30:
            rec = "MODERATE RISK: Minor sensor deviation observed. Perform routine vehicle checkup within 30 days."
            risk_level = "MODERATE"
        else:
            rec = "LOW RISK: Vehicle operational profile nominal. All powertrain components operating within factory specifications."
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
    set_scenario(4200, 94, 52, 186, 64, 45, 26, 48000)

if "unit_mode" not in st.session_state:
    st.session_state.unit_mode = "KM/H"


# -----------------------------------------------------------------------------
# SIDEBAR: SENSOR INPUTS & TELEMETRY CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    render_html("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
        <span style="font-weight:800; font-size:18px; color:#FFF;">Speed<span style="color:#22C55E;">X</span> COCKPIT</span>
        <span style="font-size:10px; font-weight:700; background:rgba(34,197,94,0.15); color:#22C55E; padding:3px 8px; border-radius:4px; border:1px solid rgba(34,197,94,0.4);">LIVE</span>
    </div>
    """)

    st.markdown("### 🏎️ MISSION PRESETS")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🟢 Highway", use_container_width=True):
            set_scenario(3200, 90, 44, 120, 35, 46, 24, 42000)
            st.rerun()
        if st.button("⚠️ Thermal", use_container_width=True):
            set_scenario(5800, 118, 78, 160, 72, 34, 38, 140000)
            st.rerun()
    with col_p2:
        if st.button("⚡ SpeedX", use_container_width=True):
            set_scenario(6500, 94, 68, 186, 64, 50, 26, 48000)
            st.rerun()
        if st.button("🚨 Critical", use_container_width=True):
            set_scenario(13200, 145, 96, 260, 98, 24, 46, 230000)
            st.rerun()

    st.markdown("<hr style='margin:14px 0; border-color:var(--card-border);'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ POWERTRAIN SENSORS")

    engine_rpm = st.slider("Engine RPM", 1000, 15000, value=int(st.session_state.engine_rpm), step=50, key="engine_rpm")
    engine_load = st.slider("Engine Load (%)", 0, 100, value=int(st.session_state.engine_load), step=1, key="engine_load")
    throttle_pos = st.slider("Throttle Position (%)", 0, 100, value=int(st.session_state.throttle_pos), step=1, key="throttle_pos")

    st.markdown("<hr style='margin:14px 0; border-color:var(--card-border);'>", unsafe_allow_html=True)
    st.markdown("### 🌡️ THERMAL & FLUIDS")

    coolant_temp = st.slider("Coolant Temperature (°C)", 60, 200, value=int(st.session_state.coolant_temp), step=1, key="coolant_temp")
    fuel_pressure = st.slider("Fuel Pressure (psi)", 20, 80, value=int(st.session_state.fuel_pressure), step=1, key="fuel_pressure")
    air_temp = st.slider("Air Temperature (°C)", -20, 60, value=int(st.session_state.air_temp), step=1, key="air_temp")

    st.markdown("<hr style='margin:14px 0; border-color:var(--card-border);'>", unsafe_allow_html=True)
    st.markdown("### 🚗 SPEED & ODOMETER")

    vehicle_speed = st.slider("Vehicle Speed (km/h)", 0, 350, value=int(st.session_state.vehicle_speed), step=1, key="vehicle_speed")
    mileage = st.number_input("Accumulated Mileage (km)", 0, 300000, value=int(st.session_state.mileage), step=1000, key="mileage")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ SYNC COCKPIT TELEMETRY", type="primary", use_container_width=True):
        st.rerun()


# Package inputs
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
# SPEEDX TOP COMMAND BAR (Matching Reference Image)
# -----------------------------------------------------------------------------
render_html("""
<div class="speedx-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <div class="speedx-logo">
            Speed<span class="speedx-logo-x">X</span>
        </div>
        <div class="speedx-pills-group">
            <span class="speedx-pill">
                <span>🛣️</span> Endless Highway Mode
            </span>
            <span class="speedx-pill">
                <span style="color:#22C55E;">📍</span> Dubai Expressway
            </span>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
        <div class="speedx-pill">
            <span>❤️</span> KM/H &bull; MPH &or;
        </div>
        <div class="speedx-pill" style="color:var(--text-muted);">
            Searched
        </div>
        <div class="speedx-icon-btn">+</div>
    </div>
</div>
""")


# -----------------------------------------------------------------------------
# MASTER COCKPIT GRID (3 COLUMNS MATCHING 1bcabb903ce3b1e3dbee4699de22c9d9.webp)
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([1.1, 1.35, 1.1])

# =============================================================================
# LEFT COLUMN: CAR PROFILE, DISTANCE & SEGMENTS
# =============================================================================
with col1:
    # Card 1: Sports Car Showcase Card (Acid Green Supercar)
    render_html(f"""
    <div class="speedx-card">
        <div class="speedx-card-title">
            <span>🏎️</span> Congred Miches // Vehicle Unit
        </div>
        <div style="text-align:center; margin:8px 0; position:relative; overflow:hidden; border-radius:12px;">
            <img src="data:image/png;base64,{CAR_B64}" style="width:100%; height:auto; display:block; border-radius:10px;" />
        </div>
        <div style="display:flex; align-items:center; justify-content:space-between; margin-top:12px; font-size:12px;">
            <span style="font-weight:700; color:#FFFFFF;">JohnX &lt; Racer</span>
            <span class="speedx-pill" style="padding:3px 10px; font-size:11px; background:rgba(34,197,94,0.12); border-color:rgba(34,197,94,0.3); color:#22C55E;">
                ★ Boost Active &lt;
            </span>
        </div>
    </div>
    """)

    # Card 2 & 3: Distance Lay & 5 Vertical Segments
    sub_col_a, sub_col_b = st.columns(2)
    with sub_col_a:
        dist_pct = min(int((mileage / 100000.0) * 100), 100)
        render_html(f"""
        <div class="speedx-card" style="padding:14px;">
            <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span style="color:#22C55E;">↻</span> Distant Lay
            </div>
            <div style="font-size:22px; font-weight:800; color:#FFF; margin:6px 0 4px 0;">
                {mileage / 1000.0:.1f} KM
            </div>
            <div class="speedx-bar-container">
                <div class="speedx-bar-fill" style="width:{dist_pct}%;"></div>
            </div>
        </div>
        """)

    with sub_col_b:
        # Calculate how many of the 5 segments are lit (1 to 5)
        num_lit = max(1, min(5, int((1.0 - risk_score) * 5) + 1))
        seg_html = "".join([f'<div class="speedx-segment-block {"lit" if i < num_lit else ""}"></div>' for i in range(5)])
        render_html(f"""
        <div class="speedx-card" style="padding:14px;">
            <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span>⏱</span> Distant Lay
            </div>
            <div style="font-size:22px; font-weight:800; color:#FFF; margin:6px 0 4px 0;">
                {mileage / 1000.0:.1f} KM
            </div>
            <div class="speedx-segments">
                {seg_html}
            </div>
        </div>
        """)

    # Distance Progress Full-Width Card
    rem_service = max(500, int((1.0 - risk_score) * 15000))
    render_html(f"""
    <div class="speedx-card">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="font-size:12px; color:var(--text-muted);">Distance Progress</span>
            <span style="font-size:16px; font-weight:700; color:#FFF;">{mileage / 1000.0:.1f} KM</span>
        </div>
        <div class="speedx-bar-container" style="height:8px; margin:8px 0 14px 0;">
            <div class="speedx-bar-fill" style="width:{min(100, int((mileage % 10000) / 100))}%;"></div>
        </div>
        <div style="border-top:1px solid var(--card-border); padding-top:10px;">
            <div style="font-size:11px; color:var(--text-muted);">Next Service Station</div>
            <div style="font-size:18px; font-weight:800; color:#22C55E; margin:2px 0;">
                In {rem_service / 1000.0:.1f}KM
            </div>
            <div style="font-size:10px; color:var(--text-dim);">Scheduled Diagnostic Routine</div>
        </div>
    </div>
    """)


# =============================================================================
# CENTER COLUMN: SPEEDX SPEEDOMETER & TELEMETRY CLUSTER
# =============================================================================
with col2:
    # Card 1: System / Fuel Rate (with glowing wave)
    fuel_pct = int(min(100, max(10, (fuel_pressure / 70.0) * 100)))
    render_html(f"""
    <div class="speedx-card" style="padding-bottom:12px;">
        <div class="speedx-card-title">
            <span>⛽</span> System // Common Rail Delivery
        </div>
        <div style="position:relative; width:100%; height:42px; overflow:hidden;">
            <svg viewBox="0 0 400 42" style="width:100%; height:100%;">
                <defs>
                    <linearGradient id="fuelWave" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#EF4444" stop-opacity="0.8"/>
                        <stop offset="50%" stop-color="#F97316" stop-opacity="0.9"/>
                        <stop offset="100%" stop-color="#EF4444" stop-opacity="0.8"/>
                    </linearGradient>
                </defs>
                <path d="M 0,22 Q 100,12 200,24 T 400,18 L 400,42 L 0,42 Z" fill="url(#fuelWave)" />
            </svg>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:4px;">
            <div>
                <span style="font-size:10px; color:var(--text-dim); text-transform:uppercase;">Fuel Rate</span>
                <div style="font-size:24px; font-weight:800; color:#FFF;">{fuel_pct}%</div>
            </div>
            <div style="text-align:right;">
                <span style="font-size:10px; color:var(--text-dim); text-transform:uppercase;">Pressure</span>
                <div style="font-size:16px; font-weight:700; color:#F97316;">{fuel_pressure} PSI</div>
            </div>
        </div>
    </div>
    """)

    # Card 2: The Core SpeedX Speedometer Gauge (Matching reference design)
    # Speed range: 0 to 350 km/h -> needle rotation angle from -130 to +130 deg
    needle_deg = -130 + (vehicle_speed / 350.0) * 260.0

    render_html(f"""
    <div class="speedx-card" style="text-align:center; padding:16px 20px 10px 20px;">
        <div style="position:relative; width:100%; max-width:340px; margin:0 auto; height:210px;">
            <svg viewBox="0 0 340 210" style="width:100%; height:100%; overflow:visible;">
                <defs>
                    <linearGradient id="speedArcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#EF4444" />
                        <stop offset="35%" stop-color="#F59E0B" />
                        <stop offset="70%" stop-color="#22C55E" />
                        <stop offset="100%" stop-color="#10B981" />
                    </linearGradient>
                    <filter id="needleGlow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="3" />
                        <feMerge>
                            <feMergeNode />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                <!-- Speedometer Arc Track -->
                <path d="M 40,165 A 130,130 0 1,1 300,165" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="12" stroke-linecap="round"/>
                <path d="M 40,165 A 130,130 0 1,1 300,165" fill="none" stroke="url(#speedArcGrad)" stroke-width="6" stroke-linecap="round"/>

                <!-- Speed Dial Numbers -->
                <text x="50" y="165" fill="#64748B" font-size="11" font-weight="700">-10</text>
                <text x="65" y="110" fill="#64748B" font-size="11" font-weight="700">40</text>
                <text x="115" y="65" fill="#CBD5E1" font-size="11" font-weight="700">180</text>
                <text x="170" y="48" fill="#CBD5E1" font-size="11" font-weight="700" text-anchor="middle">200</text>
                <text x="245" y="110" fill="#22C55E" font-size="11" font-weight="700">260</text>
                <text x="280" y="165" fill="#22C55E" font-size="11" font-weight="700">350</text>

                <!-- Center Speed Readout -->
                <text x="170" y="135" fill="#FFFFFF" font-size="52" font-weight="800" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif">{vehicle_speed}</text>
                <text x="170" y="158" fill="#94A3B8" font-size="12" font-weight="700" text-anchor="middle" letter-spacing="1">KM / H</text>

                <!-- Bottom Baseline Horizontal Bar -->
                <line x1="60" y1="180" x2="280" y2="180" stroke="#EF4444" stroke-width="3" stroke-linecap="round" filter="drop-shadow(0 0 6px #EF4444)"/>

                <!-- Rotating High-Speed Needle -->
                <g transform="translate(170, 165)">
                    <g transform="rotate({needle_deg})">
                        <line x1="0" y1="0" x2="0" y2="-120" stroke="#22C55E" stroke-width="3.5" stroke-linecap="round" filter="url(#needleGlow)"/>
                        <circle cx="0" cy="0" r="7" fill="#14171E" stroke="#22C55E" stroke-width="3"/>
                    </g>
                </g>
            </svg>
        </div>
    </div>
    """)

    # Sub-Cards below Gauge (Nitro Boosts, Damage Level, Engine Temp)
    row_sub1, row_sub2 = st.columns(2)
    with row_sub1:
        render_html(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:12px;">
            <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span style="color:#F97316;">⚡</span> Nitro Boost
            </div>
            <div style="font-size:18px; font-weight:800; color:#FFF; margin-top:2px;">
                {engine_load}% <span style="font-size:11px; color:var(--text-dim); font-weight:500;">Charged</span>
            </div>
        </div>
        """)

        # Damage Level / Risk Score
        dmg_pct = int(risk_score * 100)
        c_dmg = "#EF4444" if dmg_pct > 65 else ("#F97316" if dmg_pct > 35 else "#22C55E")
        render_html(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:0;">
            <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span style="color:{c_dmg};">⚠️</span> Damage Level
            </div>
            <div style="font-size:22px; font-weight:800; color:{c_dmg}; margin-top:2px;">
                {dmg_pct}%
            </div>
        </div>
        """)

    with row_sub2:
        render_html(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:12px;">
            <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span style="color:#22C55E;">🔋</span> Dynamic Boost
            </div>
            <div style="font-size:18px; font-weight:800; color:#FFF; margin-top:2px;">
                {throttle_pos}% <span style="font-size:11px; color:var(--text-dim); font-weight:500;">Throttle</span>
            </div>
        </div>
        """)

        # Engine Head Temperature
        c_temp = "#EF4444" if coolant_temp > 115 else ("#F97316" if coolant_temp > 100 else "#22C55E")
        render_html(f"""
        <div class="speedx-card" style="padding:12px; margin-bottom:0;">
            <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span style="color:{c_temp};">🌡️</span> Engine Temp
            </div>
            <div style="font-size:22px; font-weight:800; color:#FFF; margin-top:2px;">
                {coolant_temp}°C
            </div>
        </div>
        """)


# =============================================================================
# RIGHT COLUMN: CURRENT SCORE, TRAFFIC RADAR, DRIVER COMMS
# =============================================================================
with col3:
    # Card 1: Current Score / Health Score
    health_score = int((1.0 - risk_score) * 28540)
    render_html(f"""
    <div class="speedx-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span class="speedx-card-title" style="margin-bottom:0;">
                <span style="color:#22C55E;">●</span> Current Score
            </span>
            <span style="font-size:11px; color:#22C55E; font-weight:700;">{risk_level}</span>
        </div>
        <div style="display:grid; grid-template-columns:1.2fr 0.8fr; gap:10px;">
            <div>
                <div style="font-size:26px; font-weight:800; color:#FFF;">{health_score:,}</div>
                <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">
                    Near Misses: <span style="color:#FFF; font-weight:600;">{int(anomaly_score * 30)}</span>
                </div>
            </div>
            <div style="border-left:1px solid var(--card-border); padding-left:12px;">
                <div style="font-size:26px; font-weight:800; color:#22C55E;">{failure_prob:.1%}</div>
                <div style="font-size:11px; color:var(--text-dim); margin-top:2px;">Fail Prob</div>
            </div>
        </div>
    </div>
    """)

    # Card 2: Density & Collision Distance Radar (Matching Reference Image)
    render_html(f"""
    <div class="speedx-card" style="padding:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div class="speedx-card-title" style="margin-bottom:0;">
                <span>🚗</span> Density // Collision Radar
            </div>
            <span class="speedx-pill" style="padding:2px 8px; font-size:10px; color:#22C55E;">ACTIVE</span>
        </div>
        <div style="position:relative; width:100%; border-radius:10px; overflow:hidden;">
            <img src="data:image/png;base64,{RADAR_B64}" style="width:100%; height:auto; display:block; border-radius:8px;" />
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:var(--text-dim);">
            <span>CAN-Bus Latency: <strong style="color:#FFF;">14ms</strong></span>
            <span>Density: <strong style="color:#22C55E;">High</strong></span>
        </div>
    </div>
    """)

    # Card 3: Driver Comms & AI Maintenance Advisory
    render_html(f"""
    <div class="speedx-card">
        <div class="speedx-card-title">
            <span>📻</span> Driver Comms // AI Advisory
        </div>
        <div class="speedx-quote-box">
            <span style="color:#22C55E; font-weight:700;">"</span>{recommendation}<span style="color:#22C55E; font-weight:700;">"</span>
        </div>
        <div style="font-size:12px; color:var(--text-dim); margin:8px 0; display:flex; align-items:center; gap:6px;">
            <span>⦿</span> Nitro ready. Use wisely.
        </div>
        <div style="display:flex; align-items:center; justify-content:space-between; border-top:1px solid var(--card-border); padding-top:10px; font-size:11px;">
            <span style="color:#22C55E; font-weight:600;">✔ Gab grenes if parpools</span>
            <span style="color:var(--text-dim);">Revert &bull; Mert</span>
        </div>
    </div>
    """)


# -----------------------------------------------------------------------------
# DETAILED SPECIFICATION TABS (Preserving All Required Tables & Codes)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

tab_matrix, tab_dtc = st.tabs([
    "📋 SENSOR MATRIX & FACTORY THRESHOLDS",
    "🔧 ACTIVE DIAGNOSTIC CODES (DTC)"
])

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
        ("Coolant_Temperature", coolant_temp, "85 – 100", "°C", "Cooling loop thermal balance"),
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
        hex_color = "#22C55E" if s_color == "green" else ("#F97316" if s_color == "amber" else "#EF4444")
        val_repr = f"{p_val:,}" if isinstance(p_val, int) else f"{p_val}"
        rows_html += f'<tr><td style="font-weight:700; color:#FFFFFF;">{p_name}</td><td style="color:{hex_color}; font-weight:700;">{val_repr}</td><td style="color:var(--text-muted);">{p_opt}</td><td style="color:var(--text-dim);">{p_unit}</td><td style="color:var(--text-muted);">{p_desc}</td><td><span style="background:rgba(255,255,255,0.06); color:{hex_color}; padding:3px 8px; border-radius:4px; font-size:10px; font-weight:700;">{s_tag}</span></td></tr>'

    render_html(f"""
    <div class="speedx-card">
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>SENSOR PARAMETER</th>
                    <th>LIVE VALUE</th>
                    <th>OPTIMAL RANGE</th>
                    <th>UNIT</th>
                    <th>DIAGNOSTIC DESCRIPTION</th>
                    <th>STATUS</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """)

with tab_dtc:
    dtc_records = []
    if coolant_temp > 115:
        dtc_records.append(("P0217", "Engine Coolant Over Temperature Condition", "Cooling Loop", "CRITICAL", "Inspect radiator fan relay and cooling fluid flow."))
    elif coolant_temp > 102:
        dtc_records.append(("P0117", "Coolant Temp Sensor Circuit Low", "Sensors", "WARNING", "Inspect coolant wiring harness resistance."))

    if fuel_pressure < 34:
        dtc_records.append(("P0087", "Fuel Rail/System Pressure Too Low", "Fuel Injection", "CRITICAL", "High-pressure fuel pump delivery deficit."))
    elif fuel_pressure > 65:
        dtc_records.append(("P0088", "Fuel Rail/System Pressure Too High", "Fuel Delivery", "WARNING", "Fuel pressure regulator stuck closed."))

    if engine_rpm > 9200:
        dtc_records.append(("P0219", "Engine Overspeed Condition", "Powertrain", "CRITICAL", "Rotational speed exceeded factory limit."))

    if not dtc_records:
        dtc_records.append(("P0000", "No Active Fault Codes in ECU Memory", "Powertrain", "NOMINAL", "All sensor metrics within certified factory thresholds."))

    dtc_rows = ""
    for code, desc, sys_name, sev, action in dtc_records:
        c_sev = "#22C55E" if sev == "NOMINAL" else ("#F97316" if sev == "WARNING" else "#EF4444")
        dtc_rows += f'<tr><td style="font-family:\'JetBrains Mono\'; font-weight:700; color:{c_sev};">{code}</td><td style="font-weight:600; color:#FFF;">{desc}</td><td style="color:var(--text-muted);">{sys_name}</td><td><span style="color:{c_sev}; font-weight:700;">{sev}</span></td><td style="color:var(--text-dim);">{action}</td></tr>'

    render_html(f"""
    <div class="speedx-card">
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>DTC CODE</th>
                    <th>FAULT DESCRIPTION</th>
                    <th>SYSTEM</th>
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

# SpeedX Footer
render_html("""
<div style="text-align:center; margin-top:24px; margin-bottom:12px; font-size:11px; color:var(--text-dim);">
    SPEEDX TELEMETRY COCKPIT &bull; VERSION 4.0.0 &bull; VEHICLE PREDICTIVE MAINTENANCE ENGINE
</div>
""")
