from fastapi import FastAPI, File, UploadFile

import numpy as np
import tensorflow as tf
from PIL import Image



app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file to a local directory (optional)
    with open(file.filename, "wb") as image_file:
        image_file.write(file.file.read())

    # Call your prediction function
    result = predict_image(file.filename)

    return {"result": result}


def predict_image(image_path):
    print("Called")
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="GL_model-8808088694067036160_tflite_2024-01-21T15_44_28.861186Z_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on input image using PIL
    input_shape = input_details[0]['shape']
    # Load and resize test image
    img = Image.open(image_path).resize((input_shape[1], input_shape[2]))
    input_data = np.array(img, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    scale, zero_point = output_details[0]['quantization']
    output_data = scale * (output_data - zero_point)

    # Apply softmax to get probabilities.
    output_data = tf.nn.softmax(output_data)

    # Convert probabilities to percentages.
    output_data = output_data.numpy()[0] * 100
    if output_data[0] > output_data[1]:
        output_data = f"No Glaucoma: {output_data[0]:.2f}%"
    else:
        output_data = f"Glaucoma: {output_data[1]:.2f}%"
    print("Completed")
    return output_data

