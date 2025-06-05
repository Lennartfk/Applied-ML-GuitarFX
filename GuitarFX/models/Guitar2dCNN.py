import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.callbacks import Callback
from typing import Optional, List
import pickle
import os
import kerastuner as kt  # make sure kerastuner is installed

class GuitarEffectCNN:
    def __init__(self, num_classes, input_shape=(128, 128, 1), label_smoothing=0.1):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.label_smoothing = label_smoothing
        self.model = None
        self.history = None
        self.tuner = None
        self.best_hp = None

    def build_model(self, hp=None):
        """
        Build the CNN model.
        If hp (HyperParameters) is provided, use it for hyperparameter tuning.
        """
        if hp is None:
            # Use default parameters
            conv1_filters = 32
            conv2_filters = 64
            conv3_filters = 128
            dense_units = 256
            dropout1 = 0.25
            dropout2 = 0.25
            dropout3 = 0.25
            dropout_dense = 0.5
            lr = 1e-3
        else:
            # Use hyperparameters from tuner
            conv1_filters = hp.Int('conv1_filters', 16, 64, step=16, default=32)
            conv2_filters = hp.Int('conv2_filters', 32, 128, step=32, default=64)
            conv3_filters = hp.Int('conv3_filters', 64, 256, step=64, default=128)
            dense_units = hp.Int('dense_units', 128, 512, step=64, default=256)
            dropout1 = hp.Float('dropout1', 0.1, 0.5, step=0.1, default=0.25)
            dropout2 = hp.Float('dropout2', 0.1, 0.5, step=0.1, default=0.25)
            dropout3 = hp.Float('dropout3', 0.1, 0.5, step=0.1, default=0.25)
            dropout_dense = hp.Float('dropout_dense', 0.3, 0.7, step=0.1, default=0.5)
            lr = hp.Float('learning_rate', 1e-5, 1e-2, sampling='log', default=1e-3)

        model = models.Sequential([
            layers.Conv2D(conv1_filters, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout1),

            layers.Conv2D(conv2_filters, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout2),

            layers.Conv2D(conv3_filters, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout3),

            layers.Flatten(),
            layers.Dense(dense_units, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_dense),
            layers.Dense(self.num_classes, activation='sigmoid', dtype='float32')
        ])

        optimizer = AdamW(learning_rate=lr, weight_decay=1e-4)
        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )

        return model

    def train(self, train_dataset, val_dataset=None, epochs=30, batch_size=32,
              callbacks: Optional[List[Callback]] = None, lr=1e-3):
        """
        Train the model with given datasets and hyperparameters.
        """
        if self.model is None:
            self.model = self.build_model()
        # Update optimizer learning rate if specified and no tuner was used
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        self.history = self.model.fit(
            train_dataset[0], train_dataset[1],
            validation_data=None if val_dataset is None else (val_dataset[0], val_dataset[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return self.history

    def setup_tuner(self, max_epochs=20, directory='kt_dir', project_name='guitarfx_tuning'):
        """
        Initialize the Keras Tuner with Hyperband strategy.
        """
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=3,
            directory=directory,
            project_name=project_name,
            overwrite=True
        )
        print(f"Tuner initialized with directory={directory} and project_name={project_name}")

    def search(self, train_dataset, val_dataset, epochs=20, batch_size=64, callbacks=None):
        """
        Run hyperparameter search using tuner.
        """
        if self.tuner is None:
            raise ValueError("Tuner not initialized. Call setup_tuner() first.")

        self.tuner.search(
            train_dataset[0], train_dataset[1],
            epochs=epochs,
            validation_data=val_dataset,
            batch_size=batch_size,
            callbacks=callbacks
        )
        # Store best HP
        self.best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:", self.best_hp.values)

    def retrain_best(self, train_dataset, val_dataset=None, epochs=30, batch_size=64, callbacks=None):
        """
        Build model with best HP and retrain fully.
        """
        if self.best_hp is None:
            raise ValueError("No best hyperparameters found. Run tuner search first.")
        self.model = self.build_model(self.best_hp)

        self.history = self.model.fit(
            train_dataset[0], train_dataset[1],
            validation_data=None if val_dataset is None else (val_dataset[0], val_dataset[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return self.history

    def save(self, filepath):
        """
        Save the model and optionally the training history.
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        if self.history:
            history_path = os.path.splitext(filepath)[0] + "_history.pkl"
            with open(history_path, "wb") as f:
                pickle.dump(self.history.history, f)
            print(f"Training history saved to {history_path}")

        # Save tuner results if available
        if self.tuner is not None:
            tuner_path = os.path.splitext(filepath)[0] + "_tuner.pkl"
            with open(tuner_path, "wb") as f:
                # Save only best hyperparameters (serializable dictionary)
                pickle.dump(self.best_hp.values if self.best_hp else None, f)
            print(f"Tuner best hyperparameters saved to {tuner_path}")

    def load(self, model_path):
        """
        Load model and history from files.
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        history_path = os.path.splitext(model_path)[0] + "_history.pkl"
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                self.history = pickle.load(f)
        else:
            self.history = None

        tuner_path = os.path.splitext(model_path)[0] + "_tuner.pkl"
        if os.path.exists(tuner_path):
            with open(tuner_path, "rb") as f:
                best_hp_values = pickle.load(f)
                if best_hp_values is not None:
                    from kerastuner.engine.hyperparameters import HyperParameters
                    hp = HyperParameters()
                    for k, v in best_hp_values.items():
                        # Note: This simple approach assumes values are scalar and can be assigned
                        hp.values[k] = v
                    self.best_hp = hp
                else:
                    self.best_hp = None
        else:
            self.best_hp = None

    def get_training_history(self):
        """
        Return the training history dictionary.
        """
        return self.history.history if self.history and hasattr(self.history, 'history') else self.history

    def predict(self, inputs):
        """
        Predict with the model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
        return self.model.predict(inputs)
