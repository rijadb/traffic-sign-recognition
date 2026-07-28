import streamlit as st
import tempfile
import os

from predict import predict
from class_names import CLASS_NAMES

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign Recognition")

st.write(
    "Upload a traffic sign image."
)

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:           # execute this if file is uploaded

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width='stretch'
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:

        temp_file.write(uploaded_file.read())

        temp_path = temp_file.name

    predicted_class, confidence = predict(temp_path)

    st.subheader("Prediction")

    st.success(CLASS_NAMES[predicted_class])

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    os.remove(temp_path)