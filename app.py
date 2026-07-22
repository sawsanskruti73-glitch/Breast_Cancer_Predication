import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL & SCALER
# =========================

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("breast_cancer_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# =========================
# TITLE & DESCRIPTION
# =========================

st.title("🩺 Breast Cancer Prediction App")
st.markdown("Enter the **30 feature values** below to predict whether the tumor is benign or malignant.")

# =========================
# INPUT FIELDS (Organized in Columns)
# =========================

feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

input_values = []

# Create a 3-column layout for cleaner UI
col1, col2, col3 = st.columns(3)

for i, name in enumerate(feature_names):
    # Distribute inputs across the 3 columns
    if i % 3 == 0:
        with col1:
            val = st.number_input(name, value=1.0, format="%.5f")
    elif i % 3 == 1:
        with col2:
            val = st.number_input(name, value=1.0, format="%.5f")
    else:
        with col3:
            val = st.number_input(name, value=1.0, format="%.5f")
    
    input_values.append(val)

# =========================
# PREDICTION LOGIC
# =========================

st.markdown("---")

if st.button("Predict Cancer Type", type="primary", use_container_width=True):
    # Prepare input array
    input_data = np.array([input_values])
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data_scaled)
    predicted_value = prediction[0][0]
    
    # Display results
    st.subheader("Prediction Result:")
    
    if predicted_value > 0.5:
        st.error(f"⚠️ Dangerous (Malignant Cancer) — Score: {predicted_value:.4f}")
    else:
        st.success(f"✅ Safe (Benign Cancer) — Score: {predicted_value:.4f}")
