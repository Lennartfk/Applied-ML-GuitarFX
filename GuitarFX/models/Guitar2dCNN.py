import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from GuitarFX.features.baseline_features import FeatureExtractor

# Define parameters
INPUT_HEIGHT = 128   # number of mel bands
INPUT_WIDTH =  128  # number of time frames
INPUT_CHANNELS = 1   # mel-spectrogram channels

# 2D CNN model
def build_guitar_effect_cnn(num_classes, input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)):
    model = models.Sequential()
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model