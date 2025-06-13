import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from typing import Optional, List
import pickle
import os
import kerastuner as kt
import numpy as np

class GuitarEffectCNN:
    def __init__(self, num_classes, input_shape=(128, 128, 1), label_smoothing=0.0):
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
            dropout1 = 0.2
            dropout2 = 0.3
            dropout3 = 0.2
            dropout_dense = 0.3
            lr = 1e-4
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
            metrics=[
                AUC(name='auc'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )

        return model

    def train(self, train_dataset, val_dataset=None, epochs=30, batch_size=32,
              callbacks: Optional[List[Callback]] = None, early_stopping_patience: int = 5, earlystop=True):
        """
        Train the model with given datasets and hyperparameters.
        """
        if self.model is None:
            self.model = self.build_model()

        self.model.summary()

        if callbacks is None:
            callbacks = []
        if earlystop is True:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )


        self.history = self.model.fit(
            train_dataset[0], train_dataset[1],
            validation_data=None if val_dataset is None else (val_dataset[0], val_dataset[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        return self.history
    
    @staticmethod
    def kfold_training(X, y, label_names, n_splits=5, epochs=30, batch_size=32):   
        with open("models/multi_tuned_tuner.pkl", "rb") as f:
            best_hp_values = pickle.load(f)
        
        print("\nLoaded Hyperparamters:")
        for k, v in best_hp_values.items():
            print(f"    {k}: {v}")
        print("-" * 30)

        print("Starting K-Fold cross-validation training...")
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=23)
        fold = 1

        for train_index, val_index in mskf.split(X, y):
            print(f"Fold {fold}/{n_splits}")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            num_classes = y.shape[1]
            model = GuitarEffectCNN(num_classes=num_classes, label_smoothing=0.0)
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
            
            model.model = model.build_model()

            model.train(
                train_dataset=(X_train, y_train),
                val_dataset=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop]
            )

            model.save(f"models/tuned_cnn_fold{fold}.h5")

            fold += 1

    def setup_tuner(self, max_epochs=20, directory='kt_dir', project_name='guitarfx_tuning'):
        """
        Initialize the Keras Tuner with Hyperband strategy using val_auc as the objective.
        """
        print("Starting hyperparameter tuning...")
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=kt.Objective("val_auc", direction="max"),
            max_epochs=max_epochs,
            factor=3,
            directory=directory,
            project_name=project_name,
            overwrite=True
        )
        print(f"Tuner initialized with val_auc as objective. Directory={directory}, Project={project_name}")

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
    
    def predict_classes(self, inputs, threshold=0.5):
        probs = self.predict(inputs)
        return (probs > threshold).astype(int)
    
        