import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('skin_model.h5')


def predict(image):
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    
    prediction = model.predict(image)
    
    
    if prediction[0][0] < 0.5:
        label = 'benign'
    else:
        label = 'malignant'
    
    return label

examples=[["benign.jpg"], ["malignant.jpg"]]


image_input = gr.inputs.Image(shape=(150, 150))
label_output = gr.outputs.Label()

iface= gr.Interface(fn=predict, inputs=image_input, outputs=label_output, examples=examples,
             title="Identifying Skin Cancer", description="Predicts whether an image of skin is cancerous or not")

iface.launch()
