"""
EfficientNet model takes as input images of shape (224,224,3) and the input data should be in the range [0,255] as
normalization is included as part of the model.
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D


def ENBO_custom(num_classes: int = 5, learning_rate: float = 0.001) -> tf.keras.Model:
    """

    Args:
        num_classes:
        learning_rate:

    Returns:

    """
    inputs = Input(shape=(224, 224, 3))
    core = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    core.trainable = False
    x = GlobalAveragePooling2D(name="avg_pool")(core.output)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def unfreeze_model(model, N_unfreeze=20, LR=1e-5):
    """

    Args:
        model:
        N_unfreeze:
        LR:

    Returns:

    """
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-N_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )


# Function for unfreezing all layers in EfficientNet-B0
def unfreeze_all_model(model, LR=1e-5):
    """

    Args:
        model:
        LR:

    Returns:

    """
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
