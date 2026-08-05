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
    "Upload a traffic sign image and compare predictions from two deep learning models:"
)
st.markdown("""
-  **Custom CNN**
-  **Pretrained ResNet18 (Transfer Learning)**
""")

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=300
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name



    cnn_class, cnn_confidence = predict(temp_path)
    resnet_class, resnet_confidence = predict_resnet(temp_path)



    st.subheader("📊 Model Comparison")

    results = pd.DataFrame({
        "Model": [
            " Custom CNN",
            " ResNet18"
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
            f" **Higher Confidence:** Custom CNN "
            f"({cnn_confidence * 100:.2f}%)"
        )
    elif resnet_confidence > cnn_confidence:
        st.success(
            f" **Higher Confidence:** ResNet18 "
            f"({resnet_confidence * 100:.2f}%)"
        )
    else:
        st.info(" Both models produced the same confidence.")



    if cnn_class == resnet_class:
        st.info(
            f" Both models predicted **{CLASS_NAMES[cnn_class]}**."
        )
    else:
        st.warning(
            " The models predicted different traffic signs."
        )


    st.subheader(" Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(" Custom CNN Accuracy", "97.74%")
        st.metric("Input Size", "32 × 32")

    with col2:
        st.metric(" ResNet18 Accuracy", "99.92%")
        st.metric("Input Size", "224 × 224")

    os.remove(temp_path)