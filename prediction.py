import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the VGG16 model with the specified weights file
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

# Load the Random Forest classifier
rf_classifier = joblib.load('model_random_forest_dataset2.pkl')

# Function to extract features using VGG16
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Path to the new image for prediction
new_image_path = 'NO-S-016.jpg'

# Extract features from the new image
new_image_features = extract_features(new_image_path)

# Reshape the features for prediction
prediction_features = new_image_features.reshape(1, -1)

# Predict using the Random Forest classifier
prediction = rf_classifier.predict(prediction_features)

# Display the prediction
print("Prediction:", prediction)
