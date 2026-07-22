import gradio as gr
import numpy as np
import tensorflow as tf
import pickle

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("breast_cancer_model.keras")

# =========================
# LOAD SCALER
# =========================

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_cancer(
    mean_radius,
    mean_texture,
    mean_perimeter,
    mean_area,
    mean_smoothness,
    mean_compactness,
    mean_concavity,
    mean_concave_points,
    mean_symmetry,
    mean_fractal_dimension,

    radius_error,
    texture_error,
    perimeter_error,
    area_error,
    smoothness_error,
    compactness_error,
    concavity_error,
    concave_points_error,
    symmetry_error,
    fractal_dimension_error,

    worst_radius,
    worst_texture,
    worst_perimeter,
    worst_area,
    worst_smoothness,
    worst_compactness,
    worst_concavity,
    worst_concave_points,
    worst_symmetry,
    worst_fractal_dimension
):

    input_data = np.array([[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,

        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,

        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    predicted_value = prediction[0][0]

    if predicted_value > 0.5:
        result = " Dangerous (Malignant Cancer)"
    else:
        result = " Safe (Benign Cancer)"

    return result

# =========================
# INPUTS
# =========================

inputs = [gr.Number(value=1) for _ in range(30)]

# =========================
# INTERFACE
# =========================

interface = gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction App",
    description="Enter 30 feature values to predict cancer type."
)

interface.launch()