# app.py

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Iris",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Clasificador de Especies Iris")
st.write(
    "Ingresa las medidas de la flor para predecir la especie "
    "(`setosa`, `versicolor` o `virginica`)."
)

# ---------------------------------------------------
# CARGA DE MODELOS
# ---------------------------------------------------
# Ruta de los modelos dentro del repositorio GitHub
LOGISTIC_MODEL_PATH = "model1/logistic_regression_model.pkl"
LINEAR_MODEL_PATH = "model1/lineal_regression_model.pkl"

# Cargar modelos
logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
linear_model = joblib.load(LINEAR_MODEL_PATH)

# ---------------------------------------------------
# SELECCIÓN DEL MODELO
# ---------------------------------------------------
model_option = st.selectbox(
    "Selecciona el modelo:",
    (
        "Logistic Regression",
        "Linear Regression"
    )
)

# ---------------------------------------------------
# ENTRADAS DEL USUARIO
# ---------------------------------------------------
st.subheader("📏 Medidas de la flor")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)

# ---------------------------------------------------
# DATAFRAME DE ENTRADA
# ---------------------------------------------------
input_data = pd.DataFrame({
    "sepal.length": [sepal_length],
    "sepal.width": [sepal_width],
    "petal.length": [petal_length],
    "petal.width": [petal_width]
})

# ---------------------------------------------------
# PREDICCIÓN
# ---------------------------------------------------
if st.button("🔍 Obtener Predicción"):

    # Seleccionar modelo
    if model_option == "Logistic Regression":
        model = logistic_model
    else:
        model = linear_model

    # Predicción
    prediction = model.predict(input_data)

    # Si el modelo devuelve números
    class_names = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    result = prediction[0]

    # Convertir a nombre si es numérico
    if result in class_names:
        result = class_names[result]

    st.success(f"🌼 La especie predicha es: **{result}**")
