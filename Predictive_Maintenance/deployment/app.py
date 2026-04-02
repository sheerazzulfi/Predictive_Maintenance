import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sheerazzulfi/Predictive_Maintenance", filename="best_predictive_maintainence_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Engine Maintainance Prediction App")
st.write("""
This application predicts whether an engine requires maintenance.
Get the prediction by clicking the predict button.
""")

# User input
EngineRpm = st.number_input("Rpm of the engine", min_value=0, max_value=2500, value=50, step=50)
LubOilPressure = st.number_input("Lub oil pressure", min_value=0.0, max_value=8.0, value=3.0, step=0.1)
FuelPressure = st.number_input("Fuel pressure", min_value=0.0, max_value=25.0, value=6.0, step=0.1)
CoolantPressure = st.number_input("Coolant pressure", min_value=0.0, max_value=8.0, value=2.0, step=0.1)
lubOilTemp = st.number_input("Lub oil temperature", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
CoolantTemp = st.number_input("Coolant temperature", min_value=0.0, max_value=200.0, value=70.0, step=0.1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Engine rpm": EngineRpm,
    "Lub oil pressure": LubOilPressure,
    "Fuel pressure": FuelPressure,
    "Coolant pressure": CoolantPressure,
    "lub oil temp": lubOilTemp,
    "Coolant temp": CoolantTemp
}])


if st.button("Predict result"):
    prediction = model.predict(input_data)[0]
    result = "Engine Requires Maintainance" if prediction == 1 else "Engine is healthy"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
