import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import json

# Set page configuration
st.set_page_config(page_title="Cancer Detection Using Nanoparticles", layout="wide")

# Title and description
st.title("Cancer Detection Using Nanoparticles")
st.markdown("""
This app uses machine learning models trained on the Wisconsin Breast Cancer (Diagnostic) dataset to predict whether a breast mass is malignant or benign based on nanoparticle-like features.
Upload a CSV file or enter feature values manually to get predictions.
""")

# Load models and scaler
model_dir = "models/trained_models"
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
models = {
    "SVM": joblib.load(os.path.join(model_dir, "svm_model.joblib")),
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest_model.joblib")),
    "Neural Network": joblib.load(os.path.join(model_dir, "neural_network_model.joblib"))
}

# Load performance metrics
results_path = "notebooks/results/model_performance.csv"
results_df = pd.read_csv(results_path)

# Feature names (from ucimlrepo)
feature_names = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1',
    'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
    'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2',
    'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
    'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3'
]

# Sidebar for model selection and input method
st.sidebar.header("Model and Input Selection")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
input_method = st.sidebar.radio("Input Method", ["Upload CSV", "Manual Input"])

# Prediction function
def predict_cancer(model, input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return ["Malignant" if pred == 1 else "Benign" for pred in prediction]

# Input section
st.header("Input Data for Prediction")
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with 30 features", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        if list(input_df.columns) == feature_names:
            st.write("Uploaded Data Preview:", input_df.head())
            if st.button("Predict"):
                predictions = predict_cancer(models[model_choice], input_df)
                input_df["Prediction"] = predictions
                st.write("Predictions:", input_df[["Prediction"]])
        else:
            st.error(f"CSV must have exactly these columns: {', '.join(feature_names)}")
else:
    st.subheader("Enter Feature Values")
    input_values = []
    cols = st.columns(5)
    for i, feature in enumerate(feature_names):
        with cols[i % 5]:
            value = st.number_input(feature, value=0.0, step=0.01)
            input_values.append(value)
    if st.button("Predict"):
        input_array = np.array(input_values).reshape(1, -1)
        prediction = predict_cancer(models[model_choice], input_array)[0]
        st.success(f"Prediction: **{prediction}**")

# Performance metrics section
st.header("Model Performance Metrics")
st.write("Performance of trained models on the test set:")
st.dataframe(results_df.style.format({
    "Accuracy": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "F1-Score": "{:.4f}",
    "MSE": "{:.4f}"
}))

# Performance chart section
st.header("Model Performance Comparison")
chart_option = st.selectbox("Chart Display", ["Image (PNG)", "Interactive (Plotly)"])

if chart_option == "Image (PNG)":
    chart_path = "notebooks/figures/model_performance_comparison.png"
    if os.path.exists(chart_path):
        st.image(chart_path, caption="Model Performance Comparison", use_column_width=True)
    else:
        st.error("Chart image not found. Ensure 'model_performance_comparison.png' is in notebooks/figures/.")

else:
    chart_json_path = "notebooks/results/performance_chart.json"
    if os.path.exists(chart_json_path):
        with open(chart_json_path, 'r') as f:
            chart_data = json.load(f)
        fig = go.Figure()
        for dataset in chart_data['data']['datasets']:
            fig.add_trace(go.Bar(
                x=chart_data['data']['labels'],
                y=dataset['data'],
                name=dataset['label'],
                marker_color=dataset['backgroundColor']
            ))
        fig.update_layout(
            title=chart_data['options']['plugins']['title']['text'],
            yaxis=dict(range=[0, 1], title="Score"),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Chart JSON not found. Ensure 'performance_chart.json' is in notebooks/results/.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit for the Cancer Detection Using Nanoparticles project.")