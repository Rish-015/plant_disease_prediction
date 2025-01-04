import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load and preprocess the image
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\Mystery_soul\Desktop\Files\Internship\Plant_disease_prediction_model.keras")
    return model

def model_predict(image_path, model):
    H, W, C = 224, 224, 3
    img = cv2.imread(image_path)  # Read the file and convert into an array
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Rescaling
    img = img.reshape(1, H, W, C)  # Reshaping
    predictions = model.predict(img)[0]
    prediction = np.argmax(predictions)
    confidence = predictions[prediction]
    return prediction, confidence

# Class names for predictions
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "📸 Disease Recognition"])

# Add background image for better aesthetics
page_bg_img = '''
<style>
body {
    background-image: url("https://www.transparenttextures.com/patterns/white-wall-3.png");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: green;'>Welcome to the Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style="text-align: center; font-size: 18px;">
        🌿 This application helps in diagnosing plant diseases using AI-powered predictions.<br>
        📸 Upload an image of the plant leaf to identify diseases with confidence.<br>
        🚜 Let's promote sustainable agriculture together!
        </p>
    """, unsafe_allow_html=True)

    # Display a sample image
    sample_image = Image.open(r"C:\Users\Mystery_soul\Desktop\Files\Internship\—Pngtree—natural green leaf pattern 3d_5781606.jpg")
    st.image(sample_image, use_column_width=True)

# Disease Recognition Page
elif app_mode == "📸 Disease Recognition":
    st.markdown("<h2 style='text-align: center; color: green;'>Plant Disease Recognition</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("Upload an image of the plant leaf below:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        # Save and display the uploaded image
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())
        
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("🔍 Predict Disease"):
            st.snow()  # Add visual effect
            st.write("Analyzing the image...")
            model = load_model()
            result_index, confidence = model_predict(save_path, model)

            st.success(f"🌟 Prediction: {class_names[result_index]}")
            st.info(f"🔍 Confidence Score: {confidence * 100:.2f}%")

            if confidence < 0.6:
                st.warning("⚠️ The model's confidence is low. Please try with a clearer image.")

# Footer
st.markdown("""
    <hr>
    <footer style="text-align: center;">
        Developed by <b>Agro Puthalvan Technologies</b> | Powered by TensorFlow and Streamlit
    </footer>
""", unsafe_allow_html=True)
