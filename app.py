import streamlit as st
import cv2
from model import get_prediction
import numpy as np


st.title('OCR App')
st.write('This is a simple OCR app using CRNN model')

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if st.button('Predict'):
    if uploaded_file is not None:
        # Read the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        # print(type(file_bytes))
        # Make prediction
        prediction = get_prediction(image)
        # Display the image and prediction
        image = cv2.resize(image, (400, 100))
        st.image(image)
        # Make font size larger and green
        st.markdown(f"<h1 style='text-align: center; color: green;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)