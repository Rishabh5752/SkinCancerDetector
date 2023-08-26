import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('skin_model.h5')

# Define a function to make predictions
def predict(image):
    # Preprocess the image
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction using the model
    prediction = model.predict(image)
    
    # Get the predicted class label
    if prediction[0][0] < 0.5:
        label = 'benign'
    else:
        label = 'malignant'
    
    return label

examples=[["benign.jpg"], ["malignant.jpg"]]

# Define a Gradio interface for user interaction
image_input = gr.inputs.Image(shape=(150, 150))
label_output = gr.outputs.Label()

iface= gr.Interface(fn=predict, inputs=image_input, outputs=label_output, examples=examples,
             title="Identifying Skin Cancer", description="Predicts whether an image of skin is cancerous or not")

iface.launch()