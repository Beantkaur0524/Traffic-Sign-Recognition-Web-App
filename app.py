import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from streamlit_carousel import carousel


st.set_page_config(
    page_title="Traffic Sign Recognition Web App",
    layout="wide"
)

# Define carousel items with traffic sign recognition content
carousel_items = [
    {"img": "street.jpg", "title": "Early Detection Saves Lives", "text": "AI-powered traffic sign recognition enhances road safety by detecting crucial traffic signs in real time"},
    {"img": "Signal.jpg", "title": "Real-Time Traffic Sign Analysis", "text": "Upload images to get instant AI-based traffic sign detection and accurate recognition"},
    {"img": "Stopp.jpg", "title": "Safer Roads with AI Detection", "text": "Detect stop signs and other vital traffic signals quickly with advanced AI systems for accident prevention"},
    {"img": "maxresdefault.jpg", "title": "Traffic Sign Recognition System", "text": "Utilize AI and deep learning to automatically recognize and interpret traffic signs, improving driver awareness and compliance"}
]

LABEL_MAP = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

NUM_CLASSES = 43
INPUT_SHAPE = (32, 32, 3)

@st.cache_resource
def load_cnn_model():
    model_path = "model.keras"
    return load_model(model_path)

model_cnn = load_cnn_model()

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

carousel(items=carousel_items, fade=True, container_height=600)
st.markdown("<h1 style='text-align:center;'>Traffic Sign Recognition Web App</h1>", unsafe_allow_html=True)

(tab1,) = st.tabs(["Traffic Sign Recognition Web App"])

with tab1:
    st.subheader("Upload and Analyze Traffic Sign Image")

    file = st.file_uploader("Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

    if file:
        st.image(file, caption="Uploaded Traffic sign", width=500)

        if st.button("Analyze Traffic sign"):
            with st.spinner("Processing image..."):
                img = load_img(file, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
                img = img_to_array(img)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                pred = model_cnn.predict(img)
                class_id = int(np.argmax(pred))

                label_text = LABEL_MAP.get(class_id, f"Class {class_id}")

                st.success(f"Prediction: *{label_text}* (Class ID:{class_id})")

