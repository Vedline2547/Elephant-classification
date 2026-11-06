# IMPORT LIBRARIES
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense,
    Dropout, BatchNormalization
)

# DATASET PATHS

test_path = r"C:\Users\HP\Desktop\New ML\Elephant-classification\testElephant\test"
train_path = r"C:\Users\HP\Desktop\New ML\Elephant-classification\trainElephant\train"

# IMAGE PREPROCESSING

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# LOAD TRAINING DATA

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb'
)

# LOAD TESTING DATA

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=False
)
import json

class_indices = training_set.class_indices

with open("Class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)

print("Class indices saved successfully!")
# print("Training classes:", training_set.class_indices)
# print("Testing classes:", test_set.class_indices)

# # Verify label shapes
# images, labels = next(training_set)

# print("Image shape:", images.shape)
# print("Label shape:", labels.shape)
# # BUILD CNN MODEL
# model = Sequential()

# # BLOCK 1
# model.add(Conv2D(32, (3,3), activation='relu',
#                  input_shape=(224,224,3)))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))

# # BLOCK 2
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))

# # BLOCK 3
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))

# # BLOCK 4
# model.add(Conv2D(256, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))

# # FLATTENING
# model.add(Flatten())

# # DENSE HEAD
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))

# # OUTPUT LAYER: TWO CLASSES
# model.add(Dense(2, activation='softmax'))

# # COMPILE MODEL
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # MODEL SUMMARY
# # model.summary()
# # TRAIN MODEL
# history = model.fit(
#     training_set,
#     validation_data=test_set,
#     epochs=10
# )
# # Accuracy was not good enough
# Trained on 100 epochs accuracy increased to 0.789 and val accuracy to 0.75
# FINE-TUNING PRETRAINED MODELS
# 1.MOBILENETV2
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# # Load pretrained base
# base = MobileNetV2(
#     input_shape=(224,224,3),
#     include_top=False,
#     weights='imagenet'
# )
# base.trainable=False
# x = base.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(256,activation='relu')(x)
# x = Dropout(0.4)(x)
# outputs = Dense(training_set.num_classes,activation='softmax')(x)
# model_tf = Model(inputs=base.input,outputs=outputs)
# model_tf.compile(
#     optimizer='Adam',
#     loss = 'sparse_categorical_crossentropy',
#     metrics = ['accuracy']
# )
# # model_tf.summary()
# history_tf = model_tf.fit(
#     training_set,
#     validation_data = test_set,
#     epochs = 20
# )

# HYPERPARAMETER TUNING MOBILENETV2
# import keras_tuner as kt
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# # Assume training_set and test_set are your data generators (flow_from_directory, etc.)
# # KerasTuner works seamlessly with generators and validation_data

# num_classes = training_set.num_classes

# def build_model(hp):
#     base = MobileNetV2(
#         input_shape=(224, 224, 3),
#         include_top=False,
#         weights='imagenet'
#     )
#     base.trainable = False  # Start with frozen base; can add fine-tuning later

#     x = base.output
#     x = GlobalAveragePooling2D()(x)

#     dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=64)
#     x = Dense(dense_units, activation='relu')(x)

#     dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)
#     x = Dropout(dropout_rate)(x)

#     outputs = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=base.input, outputs=outputs)

#     learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
#     model.compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model

# # Use RandomSearch (or BayesianOptimization, Hyperband for faster)
# tuner = kt.RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=20,              # Number of different hyperparameter combinations to try
#     executions_per_trial=1,     # Can increase for more robust evaluation
#     directory='tuner_dir',
#     project_name='mobilenetv2_tune'
# )

# # Optional: Early stopping callback
# from tensorflow.keras.callbacks import EarlyStopping
# early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# # Search!
# tuner.search(
#     training_set,
#     validation_data=test_set,
#     epochs=30,                  # Max epochs per trial; early stopping will halt early
#     callbacks=[early_stop]
# )

# # Results
# best_model = tuner.get_best_models(num_models=1)[0]
# best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# print("Best hyperparameters:")
# print(best_hyperparameters.values)

# print("Best validation accuracy from search:", tuner.oracle.get_best_trials(1)[0].score)
# #Get the best single model (already loaded with its best weights)
# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.save_weights(r'C:\Users\HP\Desktop\New ML\Elephant-classificationbest_mobilenetv2.weights.h5')

# XCEPTION
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
base = Xception(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base.trainable=False
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(training_set.num_classes,activation='softmax')(x)
model_xc = Model(inputs=base.input,outputs=outputs)
model_xc.compile(
    optimize='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_xc.summary()
training_set_xc = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb'
)

# LOAD TESTING DATA

test_set_xc = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=False
)
# history_xc = model_xc.fit(
#     training_set_xc,
#     validation_data = test_set_xc,
#     epochs=20
# )
