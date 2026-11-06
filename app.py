import streamlit as st 
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing import image
st.set_page_config(page_title="Elephant Classifier",layout="centered")
st.title("Elephant Classifier")
st.write("Upload an Image-the model will predict **African** or **Asian** elephants")
# Load model and labels
@st.cache_resource
def load_arifacts():
    num_classes = 2
    base = MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable=False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=base.inputs,outputs=outputs)
    # Load the best weights
    model.load_weights(r"C:\Users\HP\Desktop\New ML\Elephant-classification\Elephant-classificationbest_mobilenetv2.weights.h5")
    # Load class indices
    with open("Class_indices.json") as f:
        class_indices = json.load(f)
    # Invert mapping
    idx_to_class = {v : k for k,v in class_indices.items()}
    return model,idx_to_class
model,idx_to_class = load_arifacts()
uploaded_file = st.file_uploader("Choose an elephant image",type=['jpg','jpeg','png'])
if uploaded_file is not None:
    st.image(uploaded_file,caption="Uploaded Image",use_container_width=True)
    # Preprocess
    img = image.load_img(uploaded_file,target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x,axis=0)
    # Predict
    preds = model.predict(x)
    pred_idx = np.argmax(preds,axis=1)[0]
    pred_class = idx_to_class[pred_idx]
    confidence = float(np.max(preds)) * 100
    st.subheader(f"Prediction : **{pred_class}**")
    st.write(f"Confidence (accuracy for this image) : **{confidence:.2f}%**")
    