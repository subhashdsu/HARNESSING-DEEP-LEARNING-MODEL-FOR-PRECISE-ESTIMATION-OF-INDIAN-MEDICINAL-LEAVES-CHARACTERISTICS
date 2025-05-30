import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the VGG16 model with pre-trained ImageNet weights
base_model = VGG16(weights=None, include_top=False)
base_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Use the VGG16 model without the top layers
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

# Function to extract features using VGG16
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg_model.predict(x)
    return features.flatten()

# Dataset directory
base_dir = 'dataset2'

# Extract features and labels from the dataset
X = []
y = []
for subdir in tqdm(os.listdir(base_dir), desc="Processing images", unit="dir"):
    subdir_path = os.path.join(base_dir, subdir)
    for img_name in os.listdir(subdir_path):
        img_path = os.path.join(subdir_path, img_name)
        features = extract_features(img_path)
        X.append(features)
        y.append(subdir)  # Assuming subdir is the class label

X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the Random Forest classifier
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)

# Save the trained Random Forest model
joblib.dump(rf_classifier, 'model_random_forest_dataset2.pkl')
