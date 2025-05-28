import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from GuitarFX.features.baseline_features import FeatureExtractor
from tensorflow.keras.optimizers import Adam


class GuitarEffectCNN():
    def __init__(self, num_classes, input_shape=(128, 128, 1)):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

    def train(self, train_dataset, val_dataset=None, epochs=10, learning_rate=1e-3):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            x=train_dataset[0],
            y=train_dataset[1],
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=32
        )

        return history

    def save(self, filepath):
        self.model.save(filepath)

    def predict(self, inputs):
        return self.model.predict(inputs)
