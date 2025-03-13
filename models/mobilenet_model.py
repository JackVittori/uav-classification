
"""MobileNetV2 support any input size greater than 32 x 32. Data has to be normalized between [-1,1] if
`include_preprocessing = True` and between [-1,1] if it is False """
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import ops
def MNV2_custom(num_classes: int = 5, learning_rate: float = 0.001):
    inputs = Input(shape=(224, 224, 3))
    inputs = ops.cast(inputs, "float32")
    inputs = tf.keras.applications.mobilenet.preprocess_input(inputs)
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
    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
