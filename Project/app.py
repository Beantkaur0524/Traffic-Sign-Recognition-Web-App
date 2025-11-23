import numpy as np
import streamlit as st
import pandas as pd
import pickle
from streamlit_carousel import carousel
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input

st.set_page_config(
    page_title="Traffic Sgn Recognition Web App",
    layout="wide"
)

# Define carousel items with traffic sign recognition content
carousel_items = [
    {"img": "street.jpg", "title": "Early Detection Saves Lives", "text": "AI-powered traffic sign recognition enhances road safety by detecting crucial traffic signs in real time"},
    {"img": "Signal.jpg", "title": "Real-Time Traffic Sign Analysis", "text": "Upload images to get instant AI-based traffic sign detection and accurate recognition"},
    {"img": "stop sign.jpg", "title": "Safer Roads with AI Detection", "text": "Detect stop signs and other vital traffic signals quickly with advanced AI systems for accident prevention"},
    {"img": "maxresdefault.jpg", "title": "Traffic Sign Recognition System", "text": "Utilize AI and deep learning to automatically recognize and interpret traffic signs, improving driver awareness and compliance"}
]

from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_cnn_model():
    model_path = "model (1).keras"
    return load_model(model_path)



model_cnn = load_cnn_model()
st.markdown("""
<style>
.block-container, .stApp, .main, .css-1outpf7 { 
    max-width: 100% !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* Ensure body and html take full width */
html, body {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Make the carousel full viewport width using the common "negative margin" trick */
.slick-slider, .streamlit-carousel, .streamlit-carousel * {
    box-sizing: border-box;
}

/* Target the slick container used by streamlit_carousel */
.slick-slider {
    position: relative !important;
    left: 50% !important;
    right: 50% !important;
    margin-left: -50vw !important;
    margin-right: -50vw !important;
    width: 100vw !important;
    max-width: 100vw !important;
    z-index: 0;
}

/* Make the inner list and track full width and keep height */
.slick-list, .slick-track {
    width: 100vw !important;
    max-width: 100vw !important;
    overflow: hidden !important;
    height: 600px !important; /* change height if you want a different hero height */
}

/* Force each slide image to fill the slide */
.slick-slide img {
    width: 100vw !important;
    height: 600px !important;  /* keep same height as above */
    object-fit: cover !important; /* cover keeps aspect ratio and fills */
    display: block;
}

/* If titles/text appear inside slides, keep them visible */
.slick-slide .slick-slide-title, .slick-slide .slick-slide-text {
    z-index: 2 !important;
}

/* Remove any default page padding on top so hero touches the top (optional) */
.block-container .element-container:first-child { padding-top: 0 !important; }

.stButton>button {
    background-color: #0077b6 !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #00b4d8;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00b4d8 !important;
    color: black !important;
}
div[data-testid="stTabs"] button {
    font-size: 18px;
}

h2.slick-slide-title {
    color: #ff1744 !important;
}
p.slick-slide-text {
    color: #009688 !important;
}
</style>
""", unsafe_allow_html=True)

carousel(items=carousel_items, fade=True, container_height=600)
st.markdown("<h1 style='text-align:center;'>Traffic Sign Recognition Web App</h1>", unsafe_allow_html=True)



(tab1,) = st.tabs(["Traffic Sign Recognition Web App"])



CLASS_NAMES = ["Stop","Turn right ahead","Turn left ahead","Go straight or right","Go straight or left", "Keep right", "Keep left","Pedestrians","Dangerous curve left'","Dangerous curve right","No vehicles","No entry","Children crossing","No passing","Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing veh over 3.5 tons","Right-of-way at intersection","Priority road","Yield","Veh > 3.5 tons prohibited","General caution","Dangerous curve left","Dangerous curve right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End speed + passing limits","Ahead only","Roundabout mandatory","End of no passing","End no passing veh > 3.5 tons"]

with tab1:
    st.subheader("Upload and Analyze Traffic Sign Image")

    file = st.file_uploader("Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

    if file:
        st.image(file, caption="Uploaded Traffic sign", width=400)

        if st.button("Analyze Traffic sign"):
            with st.spinner("Processing image..."):
                img = load_img(file, target_size=(32, 32), color_mode="rgb")
                img = img_to_array(img)
                img = img / 255.0
                img = np.expand_dims(img, 0)

                pred = model_cnn.predict(img)
                class_id = np.argmax(pred)
                confidence = np.max(pred) * 100

                st.success(f"Prediction: *{CLASS_NAMES[class_id]}*")
                st.info(f"Confidence: {confidence:.2f}%")

