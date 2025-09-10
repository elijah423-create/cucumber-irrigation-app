import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
plant_length_model = joblib.load('plant_length_model.pkl')
NOL_model = joblib.load('NOL_model.pkl')
Stem_Girth_model = joblib.load('Stem_Girth_model.pkl')

st.title("Cucumber Growth Prediction App")

st.write("Enter irrigation details to predict plant length, number of leaves, and stem girth.")

# Input boxes
irrigation_depth = st.selectbox("Irrigation Depth (% ETc)", [70, 85, 100])
irrigation_interval = st.slider("Irrigation Interval (days)", 1, 3, 2)
wap = st.slider("Weeks After Planting (WAP)", 4, 9, 6)

# Convert user input to NumPy array
input_data = np.array([[irrigation_depth, irrigation_interval, wap]])

# Predict all three outcomes
if st.button("Predict All"):
    plant_length_pred = plant_length_model.predict(input_data)[0]
    no_of_leaves_pred = NOL_model.predict(input_data)[0]
    stem_girth_pred = Stem_Girth_model.predict(input_data)[0]

    st.subheader("Predictions")
    st.write(f"Plant Length: {plant_length_pred:.2f} cm")
    st.write(f"No of Leaves: {no_of_leaves_pred:.2f}")
    st.write(f"Stem Girth: {stem_girth_pred:.2f} cm")

    # Optional bar chart of all three predictions
    fig, ax = plt.subplots()
    ax.bar(["Plant Length (cm)", "No of Leaves", "Stem Girth (cm)"],
           [plant_length_pred, no_of_leaves_pred, stem_girth_pred])
    ax.set_ylabel("Predicted Value")
    st.pyplot(fig)
