import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the PK file
from sklearn.preprocessing import StandardScaler

# Load your trained model
# model = joblib.load(r"W:\courses\SIC\assigments&tasks\20-final project\model\model.pk")
# Load your trained model (using a relative path)
model = joblib.load('model/model.pk')  # Adjust this according to your directory structure


# Streamlit UI
st.title("Epileptic Seizure Recognition")

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload CSV file with 178 features", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file
    input_data = pd.read_csv(uploaded_file)

    # Ensure the input data has the correct shape
    if input_data.shape[1] != 178:
        st.error("Input CSV must have exactly 178 features.")
    else:
        st.write("Input data:")
        st.write(input_data)

        # Create a submit button
        if st.button("Submit"):

            y_pred = model.predict(input_data)
            y_pred_classes = (y_pred > 0.5).astype(int)

            # Display results with color based on prediction
            if y_pred_classes == 1:
                st.markdown(
                    f"<span style='color:red;'>Prediction: Seizure</span>",
                    unsafe_allow_html=True,
                )
                st.image(
                    "seizure_image.jpeg", caption="Seizure Detected", width=300
                )  # Set the width to 300 pixels
            else:
                st.markdown(
                    f"<span style='color:green;'>Prediction: No Seizure</span>",
                    unsafe_allow_html=True,
                )
                st.image(
                    "no_seizure_image.jpeg", caption="No Seizure Detected", width=300
                )  # Set the width to 300 pixels
