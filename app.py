import streamlit as st
import tensorflow as tf

import numpy as np
from PIL import Image
import cv2
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the trained model

def load_trained_model():
    return tf.keras.models.load_model('braintumor.h5')  # Make sure the path is correct

# Preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (150, 150))  # Adjust size based on your model's input size
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Main function for the Streamlit app
def main():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f0f0;  /* Light gray background */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Brain Tumor Classification")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load the model
        model = load_trained_model()

        # Preprocess the image
        processed_image = preprocess_image(image)
        labels=["Glioma tumor","Meningioma tumor", "No tumor", "Pituitary tumor"]

        # Make prediction
        prediction = model.predict(processed_image)
        prediction_class = np.argmax(prediction, axis=1)[0]  # Adjust based on your model's output

        # Display the prediction
        st.write(f"Prediction: {labels[prediction_class]}")  # Adjust based on your classes

if __name__ == '__main__':
    main()
