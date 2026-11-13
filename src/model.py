"""mini-XCEPTION (Arriaga et al., 2017): a ~58k parameter CNN for 48x48 faces.

Residual blocks of depthwise-separable convolutions followed by global average
pooling, so there are no fully connected layers. This keeps inference fast
enough for real-time use on a CPU.
"""
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from .config import IMG_SIZE, NUM_CLASSES

def build_mini_xception(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES, l2=0.01):
    reg = regularizers.l2(l2)
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, use_bias=False, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(8, 3, use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for filters in (16, 32, 64, 128):
        residual = layers.Conv2D(filters, 1, strides=2, padding="same", use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False,
                                   depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False,
                                   depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Add()([x, residual])

    x = layers.Conv2D(num_classes, 3, padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="mini_xception")


def feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Model mapping a face to the 128-d pooled output of the last residual block."""
    last_add = [l for l in model.layers if isinstance(l, layers.Add)][-1]
    feats = layers.GlobalAveragePooling2D(name="embedding_gap")(last_add.output)
    return tf.keras.Model(model.input, feats, name="mini_xception_features")
