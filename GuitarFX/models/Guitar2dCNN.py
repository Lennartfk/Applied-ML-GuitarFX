import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from GuitarFX.features.baseline_features import FeatureExtractor
from tensorflow_addons.optimizers import AdamW

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
            layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])

    def train(self, train_dataset, val_dataset=None, epochs=30, learning_rate=1e-3):
        self.model.compile(
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            train_dataset[0], train_dataset[1],
            validation_data=val_dataset if val_dataset is None else (val_dataset[0], val_dataset[1]),
            epochs=epochs,
        )


        return history

    def save(self, filepath):
        self.model.save(filepath)

    def predict(self, inputs):
        return self.model.predict(inputs)
