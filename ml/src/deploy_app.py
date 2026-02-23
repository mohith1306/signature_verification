import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("models/signature_cnn.h5")

st.title("Signature Verification System")
sig1 = st.file_uploader("Upload Signature 1")
sig2 = st.file_uploader("Upload Signature 2")

if sig1 and sig2:
    # Preprocess and compare
    st.success("Prediction: Genuine / Forged")
