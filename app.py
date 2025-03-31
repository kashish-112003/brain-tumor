import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import time

def preprocess_image(image, image_size):
    img = np.array(image.convert("RGB"))  # Convert to RGB to remove alpha (if present)
    img = cv2.resize(img, (image_size, image_size))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    return img


def predict_image(model, image, class_names):
    preprocessed_img = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = class_names[predicted_class_index[0]]
    return predicted_class


def loadmodel():
    model_path = './brainTumor.keras'
    model = load_model(model_path, compile = False)
    return model

with st.spinner("Loading Model...."):
    model = loadmodel()

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

st.markdown("### Welcome to my mini project")
st.title("IS IT A BRAIN TUMOR OR HEALTHY BRAIN!")
st.header("Detection of MRI Brain Tumor")
st.text("Upload a brain MRI Image to detect tumor")
    
if "photo" not in st.session_state:
    st.session_state["photo"] = "Not Done"

def change_photo_state():
    st.session_state["photo"] = "Done"

uploaded_pic = st.file_uploader("Choose an image...", on_change = change_photo_state, type=["jpg","png","jpeg"]) 

if st.session_state["photo"] == "Done":
    
    progress_bar = st.progress(0)

    for perc_completed in range(100):
        time.sleep(0.01)
        progress_bar.progress(perc_completed + 1)
        
    st.success("Photo successfully uploaded!!")
    if uploaded_pic is not None:
        test_image = Image.open(uploaded_pic)
        resized_image = test_image.resize((150, 150))
        st.image(resized_image, caption='Uploaded Image.', use_column_width=True)

else:
    st.write("This is an example image:")
    test_image = Image.open("./Example.jpg")
    resized_image = test_image.resize((150, 150))
    st.image(resized_image, caption='Example Image', use_column_width=True)

if st.button("SUBMIT"):
    st.markdown("#### CLASSIFYING......")
    preprocessed_img = preprocess_image(test_image, 150)  # Ensure correct format
    predicted_class = predict_image(model, preprocessed_img, class_names)
    st.write(f'Predicted class: {predicted_class}')


with st.expander("Contact Us"):
    st.text("Email me at 2228028@kiit.ac.in")


with st.expander("Click here to read more"):
    st.write("This project is a made for mini project")
