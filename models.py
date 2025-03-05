# -*- coding: utf-8 -*-
"""
Definition of the EfficientNet-B0 model to be used in analysing building images
for the damage level evaluation.

Created on Wed Dec 11 17:14:57 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Flatten, Rescaling, concatenate
# from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV2





# Function for defining a new EfficientNet-B0 model
def build_model(num_classes, LR=0.001):
    inputs = Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model



# Function for unfreezing the N last layers in EfficientNet-B0
def unfreeze_model(model, N_unfreeze = 20, LR=1e-5):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-N_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )



# Function for unfreezing all layers in EfficientNet-B0
def unfreeze_all_model(model, LR=1e-5):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )



# Function for defining a new MobileNet-V2 model
def build_MN_model(num_classes, LR=0.001):
    inputs = Input(shape=(224, 224, 3))
    model = MobileNetV2(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="MobileNetV2")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
