# 🐘 Elephant Species Classification Using Deep Learning

## Overview

This project implements an image classification system capable of distinguishing between **African elephants** and **Asian elephants** using Deep Learning and Transfer Learning.

The model is built using **MobileNetV2**, a convolutional neural network (CNN) architecture pretrained on ImageNet. A custom classification head is added to adapt the pretrained network to the two-class elephant classification task.

The project also includes an interactive web application developed with **Streamlit**, allowing users to upload elephant images and receive predictions.

## Features

- Classification of African and Asian elephants.
- Transfer Learning using MobileNetV2.
- Image preprocessing and prediction using TensorFlow/Keras.
- Model weight loading for inference.
- JSON-based class index mapping.
- Interactive Streamlit web application.
- Prediction confidence and probability visualisation.

## Model Architecture

The project uses MobileNetV2 as its feature extraction backbone.

The classification architecture consists of:

1. **MobileNetV2:** Pretrained convolutional base using ImageNet weights.
2. **Global Average Pooling:** Reduces the spatial dimensions of extracted features.
3. **Dense Layer:** Learns task-specific representations.
4. **Dropout:** Helps reduce overfitting.
5. **Output Layer:** Uses Softmax activation to predict one of the two elephant species.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | MobileNetV2 |
| Learning approach | Transfer Learning |
| Input image size | 224 × 224 |
| Number of classes | 2 |
| Output activation | Softmax |
| Framework | TensorFlow / Keras |

## Dataset Structure

The dataset is organised into training and testing directories, with separate folders for each elephant species.

```text
Elephant-classification/
│
├── trainElephant/
│   ├── African/
│   └── Asian/
│
├── testElephant/
│   ├── African/
│   └── Asian/
│
├── app.py
├── Class_indices.json
├── best_mobilenetv2.weights.h5
└── README.md
```

## Technologies Used

- **Python** – Main programming language.
- **TensorFlow** – Deep Learning framework.
- **Keras** – Model development and training.
- **MobileNetV2** – Pretrained CNN architecture.
- **NumPy** – Numerical operations and image arrays.
- **Pillow (PIL)** – Image loading and processing.
- **Streamlit** – Interactive web application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Elephant-classification.git
```

```bash
cd Elephant-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow streamlit numpy pillow
```

### 4. Run the application

Ensure that the following files are available in the project directory:

- `app.py`
- `Class_indices.json`
- `best_mobilenetv2.weights.h5`

Then execute:

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use the Application

1. Launch the Streamlit application.
2. Upload an elephant image in JPG, JPEG or PNG format.
3. The application preprocesses the uploaded image.
4. MobileNetV2 processes the image and generates predictions.
5. The predicted species and confidence score are displayed.
6. View the probability breakdown for both elephant classes.

## Model Prediction

The model produces two class probabilities using the Softmax activation function:

- African Elephant
- Asian Elephant

The class with the highest predicted probability is selected as the model's prediction.

**Note:** Prediction confidence is not equivalent to overall model accuracy.

## Future Improvements

- Experiment with additional CNN architectures.
- Improve generalisation through data augmentation.
- Perform systematic hyperparameter optimisation.
- Evaluate performance using a confusion matrix, precision, recall and F1-score.
- Deploy the application for public access.
- Expand the classification system to additional animal species.

## Learning Outcomes

Through this project, I explored:

- Transfer Learning and pretrained CNN architectures.
- Image preprocessing and classification.
- TensorFlow/Keras model development.
- Loading trained model weights for inference.
- Building interactive machine learning applications with Streamlit.
- Integrating a Deep Learning model into a web application.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

GitHub: [@Vedline2547](https://github.com/Vedline2547)

---

*Developed as a Computer Vision and Deep Learning project.*
