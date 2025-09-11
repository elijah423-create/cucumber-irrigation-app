import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ----------------------------
# Set path for models
# ----------------------------
MODEL_PATH = r"C:\Users\Owner\Field work2\Field work\Data 2\CropSoilDashboard"

# ----------------------------
# Load Models
# ----------------------------
def load_model(filename):
    obj = joblib.load(os.path.join(MODEL_PATH, filename))
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("features", None)
    else:
        return obj, None

models = {
    'Saturated Hydraulic Conductivity (Ksat)': load_model('ksat_decision_tree.pkl'),
    'Soil Temperature': load_model('temperature_random_forest.pkl'),
    'Thermal Conductivity': load_model('thermal_conductivity_linear_regression.pkl'),
    'Plant Length': load_model('plant_length_model.pkl'),
    'Number of Leaves': load_model('NOL_model.pkl'),
    'Stem Girth': load_model('Stem_Girth_model.pkl'),
    'Stomatal Conductance': load_model('stomatal_random_forest.pkl'),
    'Net PAR': load_model('best_rf_netpar.pkl'),
    'Cucumber Growth': None  # composite option
}

# ----------------------------
# Units for predictions
# ----------------------------
units = {
    'Soil Temperature': "°C",
    'Stomatal Conductance': "mmol/m²·s",
    'Saturated Hydraulic Conductivity (Ksat)': "cm/hr",
    'Number of Leaves': "leaves",
    'Stem Girth': "mm",
    'Plant Length': "cm",
    'Thermal Conductivity': "W/m·K",
    'Net PAR': "μmol/m²/s"
}

# ----------------------------
# Rounding rules (decimals)
# ----------------------------
rounding = {
    'Soil Temperature': 1,
    'Thermal Conductivity': 3,
    'Plant Length': 1,
    'Stem Girth': 1,
    'Stomatal Conductance': 2,
    'Saturated Hydraulic Conductivity (Ksat)': 2,
    'Net PAR': 1,
    'Number of Leaves': 0   # whole number
}

# ----------------------------
# App Title
# ----------------------------
st.title("Cucumber Growth & Soil Property Dashboard – Ogbomoso")
st.write("Select a variable to predict and enter the required parameters.")

# ----------------------------
# Select Variable
# ----------------------------
variable = st.selectbox("Choose variable to predict", list(models.keys()))

# ----------------------------
# Input Fields
# ----------------------------
inputs = {}
inputs['Irrigation Depth'] = st.selectbox("Irrigation Depth (% ETc)", [70, 85, 100])
inputs['Irrigation Interval'] = st.selectbox("Irrigation Interval (days)", [1, 2, 3])
inputs['WAP'] = st.selectbox("Weeks After Planting (WAP)", [4, 5, 6, 7, 8, 9])

if variable in ['Saturated Hydraulic Conductivity (Ksat)', 'Soil Temperature', 'Thermal Conductivity']:
    inputs['Soil Depth'] = st.selectbox("Soil Depth (cm)", ["0-5", "5-10", "10-15", "15-20"])

# ----------------------------
# Prepare Input
# ----------------------------
soil_map = {'0-5': 1, '5-10': 2, '10-15': 3, '15-20': 4}

if variable == 'Saturated Hydraulic Conductivity (Ksat)':
    input_dict = {
        "Irrigation Depth": inputs['Irrigation Depth'],
        "Irrigation Interval": inputs['Irrigation Interval'],
        "Soil Depth (cm)": soil_map[inputs['Soil Depth']]
    }
elif variable in ['Soil Temperature', 'Thermal Conductivity']:
    input_dict = {
        "Irrigation Depth": inputs['Irrigation Depth'],
        "Irrigation Interval": inputs['Irrigation Interval'],
        "Soil Depth (cm)": soil_map[inputs['Soil Depth']],
        "WAP": inputs['WAP']
    }
else:
    input_dict = {
        "Irrigation Depth": inputs['Irrigation Depth'],
        "Irrigation Interval": inputs['Irrigation Interval'],
        "WAP": inputs['WAP']
    }

# ----------------------------
# Prediction
# ----------------------------
predictions = {}

if st.button("Predict"):
    try:
        if variable == "Cucumber Growth":
            sub_vars = ["Number of Leaves", "Stem Girth", "Plant Length"]
            for sub in sub_vars:
                model, feature_names = models[sub]
                if feature_names:
                    input_df = pd.DataFrame([input_dict])[feature_names]
                    pred = model.predict(input_df)[0]
                else:
                    input_array = np.array([list(input_dict.values())])
                    pred = model.predict(input_array)[0]

                if rounding[sub] == 0:
                    display_val = f"{int(round(pred))} {units[sub]}"
                else:
                    display_val = f"{round(pred, rounding[sub])} {units[sub]}"

                st.subheader(f"{sub}:")
                st.write(display_val)
                predictions[sub] = pred
        else:
            model, feature_names = models[variable]
            if feature_names:
                input_df = pd.DataFrame([input_dict])[feature_names]
                prediction = model.predict(input_df)
            else:
                input_array = np.array([list(input_dict.values())])
                prediction = model.predict(input_array)

            pred_value = prediction[0]
            predictions[variable] = pred_value

            if rounding[variable] == 0:
                display_val = f"{int(round(pred_value))} {units[variable]}"
            else:
                display_val = f"{round(pred_value, rounding[variable])} {units.get(variable, '')}"

            st.subheader(f"Predicted {variable}:")
            st.write(display_val)

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# ----------------------------
# Optional Chart
# ----------------------------
if st.button("Show Prediction Chart"):
    if variable == "Cucumber Growth" and predictions:
        fig, ax = plt.subplots()
        ax.bar(predictions.keys(),
               [round(predictions[sub], rounding[sub]) for sub in predictions],
               color=['lightblue', 'lightgreen', 'salmon'])
        ax.set_ylabel("Values")
        st.pyplot(fig)
    elif variable in predictions:
        fig, ax = plt.subplots()
        ax.bar([variable], [round(predictions[variable], rounding[variable])], color='lightgreen')
        ax.set_ylabel(f"{variable} ({units.get(variable, '')})")
        st.pyplot(fig)
