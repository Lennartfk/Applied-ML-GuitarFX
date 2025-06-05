import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import AdamW

class GuitarEffectCNN():
    def __init__(self, num_classes, input_shape=(128, 128, 1), label_smoothing=0.1):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(num_classes, activation='sigmoid', dtype='float32')
        ])

        self.model.summary()
        self.history = None
        self.label_smoothing = label_smoothing

    def train(self, train_dataset, val_dataset=None, epochs=30, learning_rate=0.1, batch_size=32, callbacks=None):
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )

        if callbacks is None:
            callbacks = []

        self.history = self.model.fit(
            train_dataset[0], train_dataset[1],
            validation_data=None if val_dataset is None else (val_dataset[0], val_dataset[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return self.history