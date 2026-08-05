import streamlit as st
import tempfile
import os
import pandas as pd

from predict import predict
from predict_resnet import predict as predict_resnet
from class_names import CLASS_NAMES


st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign Recognition using Deep Learning")

st.write(
    "Compare predictions from a **Custom CNN** and a **pretrained ResNet18** model."
)


st.subheader(" Models Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CNN Accuracy", "97.74%")

with col2:
    st.metric("CNN Input", "32×32")

with col3:
    st.metric("ResNet18 Accuracy", "99.92%")

with col4:
    st.metric("ResNet18 Input", "224×224")

st.divider()


st.subheader("Choose Input Method")

tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera (Experimental)"])

image_source = None
caption = ""

with tab1:

    uploaded_file = st.file_uploader(
        "Choose a traffic sign image",
        type=["jpg", "jpeg", "png"],
        key="upload"
    )

    if uploaded_file is not None:
        image_source = uploaded_file
        caption = "Uploaded Image"

with tab2:

    st.info(
    "Experimental feature. Camera predictions may be less accurate than uploaded images "
    "because the models were trained on the GTSRB dataset under controlled conditions."
)

    camera_image = st.camera_input(
        "Take a picture",
        key="camera"
    )

    if camera_image is not None:
        image_source = camera_image
        caption = "Captured Image"


if image_source is not None:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            image_source,
            caption=caption,
            width=300
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:

        temp_file.write(image_source.read())

        temp_path = temp_file.name



    cnn_class, cnn_confidence = predict(temp_path)
    resnet_class, resnet_confidence = predict_resnet(temp_path)



    st.subheader(" Prediction Comparison")

    results = pd.DataFrame({
        "Model": [
            "Custom CNN",
            "ResNet18"
        ],
        "Prediction": [
            CLASS_NAMES[cnn_class],
            CLASS_NAMES[resnet_class]
        ],
        "Confidence": [
            f"{cnn_confidence * 100:.2f}%",
            f"{resnet_confidence * 100:.2f}%"
        ]
    })

    st.table(results)



    if cnn_confidence > resnet_confidence:

        st.success(
            f" Higher Confidence: Custom CNN ({cnn_confidence * 100:.2f}%)"
        )

    elif resnet_confidence > cnn_confidence:

        st.success(
            f" Higher Confidence: ResNet18 ({resnet_confidence * 100:.2f}%)"
        )

    else:

        st.info(" Both models produced the same confidence.")


    if cnn_class == resnet_class:

        st.info(
            f" Both models predicted: {CLASS_NAMES[cnn_class]}"
        )

    else:

        st.warning(
            " The models predicted different traffic signs."
        )

    os.remove(temp_path)