import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Define parameters
INPUT_HEIGHT = 128   # number of mel bands
INPUT_WIDTH =  128  # number of time frames
INPUT_CHANNELS = 1   # mel-spectrogram channels
NUM_CLASSES = 8      # number of guitar effects

# 2D CNN model
def build_guitar_effect_cnn(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), num_classes=NUM_CLASSES):
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

if __name__ == "__main__":
    model = build_guitar_effect_cnn()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # example
    dataset = tf.data.Dataset.from_tensor_slices((mel_specs, labels))
    dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

    model.fit(dataset, epochs=30, validation_data=val_dataset)
