import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import math

# This is a dummy model to simulate the prediction behavior
# of a real trained model, as the original .pkl file was not provided.
class VehiclePredictiveMaintenanceModel:
    """
    A simulated model for vehicle predictive maintenance.
    It generates predictions based on simple rules derived from
    the input sensor data.
    """
    def __init__(self, metadata):
        self.metadata = metadata

    def predict_maintenance_needs(self, input_data):
        """
        Simulates the prediction process.
        
        Args:
            input_data (dict): A dictionary of vehicle sensor data.
        
        Returns:
            dict: A dictionary containing simulated risk scores,
                  failure probabilities, anomaly scores, and recommendations.
        """
        # Define a simplified risk calculation
        # Higher mileage, RPM, and coolant temp lead to higher risk
        mileage_norm = min(input_data['Mileage'] / 200000, 1.0)
        rpm_norm = min(input_data['Engine_RPM'] / 10000, 1.0)
        temp_norm = min(input_data['Coolant_Temperature'] / 120, 1.0)
        
        # A simple weighted average to get a risk score between 0 and 1
        risk_score = (0.5 * mileage_norm) + (0.3 * rpm_norm) + (0.2 * temp_norm)
        
        # Apply a sigmoid-like curve to the risk score to make it more realistic
        risk_score = 1 / (1 + math.exp(-10 * (risk_score - 0.5)))
        
        # Calculate a plausible anomaly score and failure probability
        anomaly_score = np.random.uniform(0.1, 0.4) + (risk_score * 0.5)
        failure_probability = risk_score * 0.9 + np.random.uniform(0.0, 0.1)

        # Determine the recommendation based on the risk score
        if risk_score > 0.7:
            recommendation = "URGENT: High risk of component failure. Do not operate the vehicle until it has been inspected."
        elif risk_score > 0.5:
            recommendation = "HIGH RISK: Schedule a maintenance appointment soon. A critical component may be at risk."
        elif risk_score > 0.3:
            recommendation = "MODERATE RISK: Vehicle health is declining. Consider a service check in the near future."
        else:
            recommendation = "LOW RISK: Vehicle health is good. Continue with routine checks and maintenance."

        return {
            'risk_scores': [risk_score],
            'failure_probabilities': [failure_probability],
            'anomaly_scores': [anomaly_score],
            'recommendations': [recommendation]
        }

def main():
    st.set_page_config(page_title="Vehicle Predictive Maintenance", layout="wide")

    st.title("🚗 Vehicle Predictive Maintenance Dashboard")
    st.markdown("Enter your vehicle's sensor data to get maintenance predictions")

    # The metadata from the provided JSON object is included here
    model_metadata = {
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

    # Initialize our dummy model
    model = VehiclePredictiveMaintenanceModel(model_metadata)

    # Sidebar for input
    st.sidebar.header("Vehicle Sensor Data")

    # Input fields
    engine_rpm = st.sidebar.slider("Engine RPM", 1000, 15000, 2500, 50)
    coolant_temp = st.sidebar.slider("Coolant Temperature (°C)", 60, 500, 90, 1)
    engine_load = st.sidebar.slider("Engine Load (%)", 0, 100, 50, 5)
    vehicle_speed = st.sidebar.slider("Vehicle Speed (km/h)", 0, 350, 60, 5)
    throttle_pos = st.sidebar.slider("Throttle Position (%)", 0, 100, 25, 5)
    fuel_pressure = st.sidebar.slider("Fuel Pressure (psi)", 20, 80, 45, 1)
    air_temp = st.sidebar.slider("Air Temperature (°C)", -20, 60, 25, 1)
    mileage = st.sidebar.number_input("Mileage (km)", 0, 300000, 50000, 1000)

    # Create input dictionary
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

    # Make prediction
    if st.sidebar.button("Analyze Vehicle Health", type="primary"):
        try:
            results = model.predict_maintenance_needs(input_data)

            # Main dashboard layout
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Risk gauge
                risk_score = results['risk_scores'][0]

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Failure Probability", f"{results['failure_probabilities'][0]:.1%}")
                st.metric("Anomaly Score", f"{results['anomaly_scores'][0]:.3f}")

            with col3:
                # The risk level logic should match the recommendation logic
                risk_level = "LOW"
                if risk_score > 0.7:
                    risk_level = "URGENT"
                elif risk_score > 0.5:
                    risk_level = "HIGH"
                elif risk_score > 0.3:
                    risk_level = "MODERATE"
                
                st.metric("Risk Level", risk_level)

            # Recommendation
            st.subheader("🔧 Maintenance Recommendation")
            recommendation = results['recommendations'][0]

            if "URGENT" in recommendation:
                st.error(recommendation)
            elif "HIGH RISK" in recommendation:
                st.warning(recommendation)
            elif "MODERATE RISK" in recommendation:
                st.info(recommendation)
            else:
                st.success(recommendation)

            # Detailed analysis
            st.subheader("📊 Detailed Analysis")

            # Input summary
            input_df = pd.DataFrame([input_data]).T
            input_df.columns = ['Value']
            input_df.index.name = 'Parameter'
            st.table(input_df)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
