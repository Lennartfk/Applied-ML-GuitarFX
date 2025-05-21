import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import mixed_precision

from GuitarFX.features.cnn_features import CNNFeatureExtractor
from GuitarFX.models.Guitar2dCNN import build_guitar_effect_cnn
from GuitarFX.metrics.metrics import ModelMetrics

# more memory efficient
mixed_precision.set_global_policy('mixed_float16')

def main():
    # Paths to your dataset folders, e.g. ["data/train", "data/val"]
    dataset_paths = [
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon\Gitarre monophon\Samples",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2\Gitarre monophon2\Samples",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon\Gitarre polyphon\Samples",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon2\Gitarre polyphon2"
    ]

    # Feature extraction
    extractor = CNNFeatureExtractor(dataset_paths)
    X, y, label_names = extractor.get_cnn_features(read_file=True, filename="cnn_mels_trimmed.npz")  # load cached if exists

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X = X[..., np.newaxis]

    # Split into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full # This would make X_val 20% of total
    )

    num_classes = len(label_encoder.classes_)

    # Build CNN model
    model = build_guitar_effect_cnn(num_classes=num_classes, input_shape=X_train.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model with validation split from training data
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # Predict on test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Evaluate metrics
    metrics = ModelMetrics(
        y_pred=y_pred_probs,
        y_actual=y_test,
        label_encoder=label_encoder,
        train_accuracy=history.history['accuracy'],
        val_accuracy=history.history['val_accuracy'],
        train_loss=history.history['loss'],
        val_loss=history.history['val_loss']
    )
    metrics.report_all_results()

if __name__ == "__main__":
    main()
