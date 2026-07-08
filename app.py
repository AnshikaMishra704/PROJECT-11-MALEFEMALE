import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Male Female Classifier",
    page_icon="🧑",
    layout="centered"
)

st.title("🧑 Male vs Female Image Classifier")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("binary_image_classifier.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((150, 150))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.success("Prediction : Male 👨")
        st.write(f"Confidence : {prediction[0][0]*100:.2f}%")
    else:
        st.success("Prediction : Female 👩")
        st.write(f"Confidence : {(1-prediction[0][0])*100:.2f}%")
