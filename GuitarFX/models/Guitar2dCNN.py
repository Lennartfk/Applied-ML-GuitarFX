import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from GuitarFX.features.baseline_features import FeatureExtractor
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.callbacks import Callback
from typing import Optional, List
import pickle
import os

class GuitarEffectCNN():
    def __init__(self, num_classes, input_shape=(128, 128, 1), label_smoothing=0.1):
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
            layers.Dense(num_classes, activation='sigmoid', dtype='float32')
        ])

        self.model.summary()
        self.history = None
        self.label_smoothing = label_smoothing

    def train(self, train_dataset, val_dataset=None, epochs=30, lr=1e-3, batch_size=32, callbacks: Optional[List[Callback]] = None):
        
        optimizer = AdamW(learning_rate=lr, weight_decay=1e-4)

        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            train_dataset[0], train_dataset[1],
            validation_data=None if val_dataset is None else (val_dataset[0], val_dataset[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return self.history

    def save(self, filepath):
        self.model.save(filepath)
        if self.history:
            history_path = os.path.splitext(filepath)[0] + "_history.pkl"
            with open(history_path, "wb") as f:
                pickle.dump(self.history.history, f)

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        history_path = os.path.splitext(model_path)[0] + "_history.pkl"
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                self.history = pickle.load(f)
        else:
            self.history = None

    def get_training_history(self):
        return self.history.history if self.history and hasattr(self.history, 'history') else self.history
        
    def predict(self, inputs):
        return self.model.predict(inputs)