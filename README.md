# Traffic Sign Recognition using Deep Learning

This project was developed as part of a Senior Design Project and demonstrates traffic sign
classification using deep learning techniques. It includes a custom Convolutional Neural
Network (CNN) built from scratch, a ResNet18 model using transfer learning, and a Streamlit
application for image classification.

---

## Project Structure

- `dataset/` – German Traffic Sign Recognition Benchmark (GTSRB) dataset
- `models/` – Saved trained model weights
- `src/train.py` – Training script for the custom CNN
- `src/train_resnet.py` – Training script for ResNet18
- `src/data_loader.py` – Creates PyTorch DataLoaders for efficient batch loading during model training
- `src/model.py` – Custom CNN architecture
- `src/resnet_model.py` – ResNet18 architecture
- `src/dataset.py` – Dataset loader and preprocessing for CNN
- `src/dataset_resnet.py` – Dataset loader and preprocessing for ResNet18
- `src/predict.py` – Prediction script for the custom CNN
- `src/predict_resnet.py` – Prediction script for ResNet18
- `src/app.py` – Streamlit web application
- `src/class_names.py` – Traffic sign class labels
- `src/utils.py` – Helper functions

---

## Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

Download it from:

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

After downloading, place the dataset in the following directory:

```text
dataset/
└── GTSRB/
```

## Models

### Custom CNN

- Built from scratch using PyTorch
- Trained on the GTSRB dataset
- Input image size: 32×32

### ResNet18

- Pretrained on ImageNet
- Fine-tuned using transfer learning
- Retrained `layer4` and the final classification layer

---

## Dataset

The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

- 43 traffic sign classes
- More than 50,000 labeled images
- Public benchmark dataset for traffic sign classification

---

## Results

| Model      | Training Accuracy |
| ---------- | ----------------: |
| Custom CNN |            97.74% |
| ResNet18   |            99.92% |

Example prediction:

```text
Class 20 : 99.96%
Class 31 : 0.01%
Class 2  : 0.01%

Prediction:
Dangerous curve to the right

Confidence:
99.96%
```

---

## Prerequisites

- Python 3.10+
- PyTorch
- OpenCV
- Streamlit

---

## Setup Instructions

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Custom CNN

```bash
python src/train.py
```

### Train ResNet18

```bash
python src/train_resnet.py
```

### Run Predictions

Custom CNN:

```bash
python src/predict.py
```

ResNet18:

```bash
python src/predict_resnet.py
```

### Launch the Streamlit Application

```bash
streamlit run src/app.py
```

---

## Experimental Camera Mode

The application includes an experimental camera input feature using Streamlit.

The trained models were developed using the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains well-centered and controlled traffic sign images. As a result, predictions on live camera images may be less accurate due to differences in lighting, viewing angle, background clutter, and image quality.

The primary evaluation and comparison of the models is therefore based on uploaded traffic sign images.

## Key Features

- Custom CNN implemented from scratch
- ResNet18 transfer learning
- Image preprocessing with OpenCV and Torchvision
- Confidence score prediction
- Streamlit web interface
- Comparison between a custom architecture and a pretrained model

---
