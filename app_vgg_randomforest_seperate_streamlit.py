import os
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image
import io 
import tensorflow as tf

# Load the VGG16 model with the specified weights file
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
model_vgg = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

# Load the Random Forest classifier
rf_classifier = joblib.load('model_random_forest_dataset2.pkl')

# Load the trained VGG16 model for classification
model = tf.keras.models.load_model('model_vgg_dataset2.h5')

# Get the class names for the VGG16 model
class_names = sorted(os.listdir('dataset2'))  # replace with your actual directory

# Function to extract features using VGG16
def extract_features_vgg(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize the image to match VGG input
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_vgg.predict(x)
    return features.flatten()

# Function to predict using Random Forest
def predict_rf(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    new_image_path = 'uploaded_image.jpg'
    image.save(new_image_path)
    new_image_features = extract_features_vgg(new_image_path)
    prediction_features = new_image_features.reshape(1, -1)
    prediction = rf_classifier.predict(prediction_features)
    return prediction

# Function to load and prepare the image for VGG16 model
def load_and_prep_image(image, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape,, color_channels)
    """
    # Decode it into a tensor
    img = tf.image.decode_image(image)

    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img

# Streamlit app
st.title("Image Classification with VGG16 and Random Forest")
st.write("Choose a model for classification:")
model_choice = st.radio("Model", ("VGG16", "Random Forest"))

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    if model_choice == "VGG16":
        st.write("Classifying with VGG16...")
        image_vgg = load_and_prep_image(image_bytes)
        image_vgg = tf.expand_dims(image_vgg, axis=0)
        pred_vgg = model.predict(image_vgg)
        pred_class_vgg = class_names[np.argmax(pred_vgg)]
        st.write(f"Prediction: {pred_class_vgg}")
    elif model_choice == "Random Forest":
        st.write("Classifying with Random Forest...")
        prediction_rf = predict_rf(image_bytes)
        st.write(f"Prediction: {prediction_rf}")
