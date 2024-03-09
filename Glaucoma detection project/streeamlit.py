import streamlit as st
from PIL import Image
import predict
import pathlib
import textwrap
import PIL.Image

import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

import google.generativeai as genai

# Set page title and favicon
st.set_page_config(
    page_title="Image and Text Analysis App", 
    page_icon=":camera:"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: white; /* Updated color to white */
        text-align: center;
        margin-bottom: 30px;
    }
    .description {
        font-size: 18px;
        color: white; /* Updated color to white */
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction {
        font-size: 24px;
        color: white; /* Updated color to white */
        text-align: center;
        margin-top: 30px;
    }
    .text-box {
        width: 100%;
        height: 200px; /* Adjust height as needed */
        overflow-y: scroll; /* Enable vertical scrolling if text overflows */
        padding: 10px;
        background-color: #333;
        color: white;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Main Streamlit code
def main():
    # Title and description
    st.markdown("<h1 class='title'>OptiScan App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Upload an image and get predictions!</p>", unsafe_allow_html=True)

    # Placeholder for logo
    logo_image = st.sidebar.image("logo.jpg", use_column_width=True)

    # File uploader for image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Resize the uploaded image to 256x256 pixels
        image = Image.open(uploaded_image)
        image_resized = image.resize((100, 100))

        # Perform image processing
        prediction = predict.predict_image(uploaded_image)

        # Display prediction
        st.markdown("<p class='prediction'>Prediction Score: {}</p>".format(prediction), unsafe_allow_html=True)

        # Google API for text generation
        key = 'AIzaSyB2g451h6la4HEC8RjHBMYOSo9cqpaB7d0'
        GOOGLE_API_KEY = key
        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro-vision', generation_config=generation_config)

        if uploaded_image is not None:
            if "No" in prediction:  # Check if glaucoma is not detected
                response = model.generate_content(["""
                Here is the image of the fundus image, the OCD ratio is 0.4. Assume you are personal eye health advisor and give insights over this.
                1. Reasons
                2. Prevention
                3. Regular Eye Exercises
                                                   
                in numbered bullet points.  
                (Response for Non-Glaucoma)                                     
                """, image], stream=False)
            else:
                response = model.generate_content(["""
                Here is the image of the fundus image, the OCD ratio is 0.7. Assume you are personal eye health advisor and give insights over this.
                1. Reasons
                2. Prevention
                3. Regular Eye Exercises
                                                   
                in bullet points.  
                (Response for Glaucoma)                                   
                """, image], stream=False)
            st.text_area("Recommended Advice", value=response.text, height=200, key='generated-textbox')
        else:
            st.write("Please upload an image.")


if __name__ == "__main__":
    main()
