import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from PIL import Image

cnn = load_model('trained_model.h5')  # Load the saved model

def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
        
    if result[0][0] >= 0.5:
        prediction = 'Pneumonia Detected'
    else:
        prediction = 'Normal'
    
    return prediction

st.title("Pneumonia Classifier")
st.write("Upload an X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Uploaded Image:")
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    image_path = './uploaded_image.jpg'
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    prediction = predict_image(image_path)
    
    st.subheader("Prediction:")
    if prediction == 'Pneumonia Detected':
        st.error(prediction)
    else:
        st.success(prediction)
