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

# Additional data based on prediction
data = data = {
    "Alpinia Galanga (Rasna)": {
        "medicinal_properties": ["Anti-inflammatory", "Antioxidant", "Antimicrobial"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Arthritis", "Indigestion"]
    },
    "Amaranthus Viridis (Arive-Dantu)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antibacterial"],
        "geo_location": ["Worldwide"],
        "disease_curable": ["Asthma", "Diabetes", "High Blood Pressure"]
    },
    "Artocarpus Heterophyllus (Jackfruit)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Digestive Disorders", "Skin Diseases"]
    },
    "Azadirachta Indica (Neem)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antiviral"],
        "geo_location": ["India"],
        "disease_curable": ["Skin Diseases", "Malaria", "Diabetes"]
    },
    "Basella Alba (Basale)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Anticancer"],
        "geo_location": ["India"],
        "disease_curable": ["Constipation", "Anemia"]
    },
    "Brassica Juncea (Indian Mustard)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Anticancer"],
        "geo_location": ["India"],
        "disease_curable": ["Cough", "Asthma", "Bronchitis"]
    },
    "Carissa Carandas (Karanda)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antidiabetic"],
        "geo_location": ["India"],
        "disease_curable": ["Diabetes", "Wounds", "Ulcers"]
    },
    "Citrus Limon (Lemon)": {
        "medicinal_properties": ["Antioxidant", "Antibacterial", "Antiviral"],
        "geo_location": ["Worldwide"],
        "disease_curable": ["Scurvy", "Indigestion", "Skin Care"]
    },
    "Ficus Auriculata (Roxburgh fig)": {
        "medicinal_properties": ["Antioxidant", "Antidiabetic", "Antimicrobial"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Diabetes", "Skin Diseases", "Wounds"]
    },
    "Ficus Religiosa (Peepal Tree)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antidiabetic"],
        "geo_location": ["India"],
        "disease_curable": ["Asthma", "Jaundice", "Diabetes"]
    },
    "Hibiscus Rosa-sinensis": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antibacterial"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Hair Loss", "Hypertension", "Cough"]
    },
    "Jasminum (Jasmine)": {
        "medicinal_properties": ["Antidepressant", "Antiseptic", "Antispasmodic"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Anxiety", "Skin Diseases", "Menstrual Disorders"]
    },
    "Mangifera Indica (Mango)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Anticancer"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Indigestion", "Heat Stroke", "Anemia"]
    },
    "Mentha (Mint)": {
        "medicinal_properties": ["Antimicrobial", "Antispasmodic", "Digestive Aid"],
        "geo_location": ["Worldwide"],
        "disease_curable": ["Indigestion", "Nausea", "Headache"]
    },
    "Moringa Oleifera (Drumstick)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antidiabetic"],
        "geo_location": ["India", "Africa"],
        "disease_curable": ["Diabetes", "Anemia", "Malnutrition"]
    },
    "Muntingia Calabura (Jamaica Cherry-Gasagase)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Fever", "Hypertension", "Diabetes"]
    },
    "Murraya Koenigii (Curry)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["India"],
        "disease_curable": ["Diabetes", "Diarrhea", "Nausea"]
    },
    "Nerium Oleander (Oleander)": {
        "medicinal_properties": ["Cardiotonic", "Anticancer", "Antimicrobial"],
        "geo_location": ["Mediterranean Region", "Asia"],
        "disease_curable": ["Heart Diseases", "Cancer"]
    },
    "Nyctanthes Arbor-tristis (Parijata)": {
        "medicinal_properties": ["Antipyretic", "Antiarthritic", "Antioxidant"],
        "geo_location": ["India"],
        "disease_curable": ["Fever", "Arthritis", "Skin Diseases"]
    },
    "Ocimum Tenuiflorum (Tulsi)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["India"],
        "disease_curable": ["Cough", "Cold", "Insect Bites"]
    },
    "Piper Betle (Betel)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antioxidant"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Oral Health", "Digestive Disorders"]
    },
    "Plectranthus Amboinicus (Mexican Mint)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antioxidant"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Respiratory Disorders", "Digestive Disorders"]
    },
    "Pongamia Pinnata (Indian Beech)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antioxidant"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Skin Diseases", "Wounds", "Rheumatism"]
    },
    "Psidium Guajava (Guava)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Diarrhea", "Dysentery", "Skin Disorders"]
    },
    "Punica Granatum (Pomegranate)": {
        "medicinal_properties": ["Antioxidant", "Anticancer", "Antimicrobial"],
        "geo_location": ["Middle East", "India"],
        "disease_curable": ["Heart Diseases", "Diabetes", "High Blood Pressure"]
    },
    "Santalum Album (Sandalwood)": {
        "medicinal_properties": ["Antiseptic", "Anti-inflammatory", "Astringent"],
        "geo_location": ["India", "Australia"],
        "disease_curable": ["Skin Diseases", "Urinary Tract Infections"]
    },
    "Syzygium Cumini (Jamun)": {
        "medicinal_properties": ["Antidiabetic", "Antioxidant", "Antimicrobial"],
        "geo_location": ["India"],
        "disease_curable": ["Diabetes", "Digestive Disorders", "Skin Diseases"]
    },
    "Syzygium Jambos (Rose Apple)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anticancer"],
        "geo_location": ["Southeast Asia"],
        "disease_curable": ["Diabetes", "Digestive Disorders", "Cancer"]
    },
    "Tabernaemontana Divaricata (Crape Jasmine)": {
        "medicinal_properties": ["Antipyretic", "Antispasmodic", "Anti-inflammatory"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Fever", "Muscle Pain", "Inflammation"]
    },
    "Trigonella Foenum-graecum (Fenugreek)": {
        "medicinal_properties": ["Antidiabetic", "Antioxidant", "Anti-inflammatory"],
        "geo_location": ["India", "Mediterranean Region"],
        "disease_curable": ["Diabetes", "Digestive Disorders", "Skin Inflammation"]
    }
}


# Function to extract features using VGG16
def extract_features(img_path):
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
    new_image_features = extract_features(new_image_path)
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
st.title("Image Classification with VGG16 Deep Learning and Random Forest Ensemble")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Predict with VGG16
    image_vgg = load_and_prep_image(image_bytes)
    image_vgg = tf.expand_dims(image_vgg, axis=0)
    pred_vgg = model.predict(image_vgg)
    pred_class_vgg = class_names[np.argmax(pred_vgg)]

    # Predict with Random Forest
    prediction_rf = predict_rf(image_bytes)
    pred_class_rf = prediction_rf[0]

    if pred_class_vgg == pred_class_rf:
        st.write("Based on Majority Voting:")
        st.write(f"Prediction: {pred_class_vgg}")

        if pred_class_vgg in data:
            st.markdown(f"**Medicinal Properties:** {', '.join(data[pred_class_vgg]['medicinal_properties'])}")
            st.markdown(f"**Geo Location:** {', '.join(data[pred_class_vgg]['geo_location'])}")
            st.markdown(f"**Disease Curable:** {', '.join(data[pred_class_vgg]['disease_curable'])}")
        else:
            st.write("No additional data available")
    else:
        st.write("Predictions from Individual Models:")
        st.markdown(f"Prediction from Random Forest Ensemble (92% accurate): **{pred_class_rf}**")
        if pred_class_rf in data:
            st.markdown(f"**Medicinal Properties:** {', '.join(data[pred_class_rf]['medicinal_properties'])}")
            st.markdown(f"**Geo Location:** {', '.join(data[pred_class_rf]['geo_location'])}")
            st.markdown(f"**Disease Curable:** {', '.join(data[pred_class_rf]['disease_curable'])}")
        else:
            st.write("No additional data available")
        st.markdown("---")  # Horizontal ruler

        st.markdown(f"Prediction from VGG16 (81% accurate): **{pred_class_vgg}**")
        if pred_class_vgg in data:
            st.markdown(f"**Medicinal Properties:** {', '.join(data[pred_class_vgg]['medicinal_properties'])}")
            st.markdown(f"**Geo Location:** {', '.join(data[pred_class_vgg]['geo_location'])}")
            st.markdown(f"**Disease Curable:** {', '.join(data[pred_class_vgg]['disease_curable'])}")
        else:
            st.write("No additional data available")

