from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

if __name__ == '__main__':
    a = MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
