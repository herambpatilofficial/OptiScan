import gradio as gr
from PIL import Image
from test import predict_image
import os

def wrapper_func(uploaded_file):
    image = Image.fromarray(uploaded_file.astype('uint8'), 'RGB')
    image.save("temp.jpg", "JPEG")  # Save the image as a JPEG file
    output = predict_image("temp.jpg")
    os.remove("temp.jpg")  # Delete the temporary image
    return output


# Add Title and Description
title = "Glaucoma Detection"
description = "This is a simple web app to detect Glaucoma from fundus images."


iface = gr.Interface(fn=wrapper_func, inputs="image", outputs="text",title=title)
iface.launch(share=True)