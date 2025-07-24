import streamlit as st
import eda, prediction

st.sidebar.title('Navigation')
app_mode = st.sidebar.radio("Choose the App Mode", ["EDA", "Prediction"])

if app_mode == "EDA":
    eda.app()
elif app_mode == "Prediction":
    prediction.app()