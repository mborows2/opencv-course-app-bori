# Importing the libraries.
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from utils import *

def prediction(net):   
    image_data = st_canvas(
        fill_color="rgb(15, 15, 15)", 
        stroke_width = 15,
        stroke_color = "rgb(255,255,255)",
        background_color = "rgb(0,0,0)",
        height=280,
        width=280, 
        update_streamlit = True,
        key="canvas",
    )

    # Predicting the image
    if image_data is not None:
        if st.button('Predict'):
            # Model inference
            digit, confidence = predictDigit(image_data,net)
            st.write('Recognized Digit: {}'.format(digit))
            st.write('Confidence: {:.2f}'.format(confidence))

def main():
    # Load Digit Recognition model
    net = cv2.dnn.readNetFromONNX('model.onnx')
    
    st.title("Digit Recognizer")
    st.write("\n\n")
    st.write("Draw a digit below and click on Predict button")
    st.write("\n")
    st.write("To clear the digit, uncheck trashcan icon")

    prediction(net)

if __name__ == '__main__':
    main()
