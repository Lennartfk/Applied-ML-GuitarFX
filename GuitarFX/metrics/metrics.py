from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
import numpy as np


class ModelMetrics:
    """
    Reports results for a classification machine learning model.
    """
    def __init__(self,
                 y_pred: Union[np.ndarray, List[float]],
                 y_actual: Union[np.ndarray, List[float]],
                 label_encoder: LabelEncoder,
                 train_accuracy: List[float] = None,
                 val_accuracy: List[float] = None,
                 train_loss: List[float] = None,
                 val_loss: List[float] = None
                 ) -> None:
        """
        Initializes the attributes of the ModelMetrics class that contain
        information about the performance of the model.

        Inputs:
            y_pred: A vector of a list of probabilities or predictions
            for each class. The index of each vector of the y_pred corresponds
            to the index of the vector for each test features index.
            y_actual: The vector of ground truths for each index
            of features in the test vector.
            label_encoder (LabelEncoder): Label encoder that encodes labels
            from textual to integer values.
            train_accuracy (List[float]): The training accuracy over each
            epoch.
            val_accuracy (List[float]): The validation accuracy over each
            epoch.
            train_loss (List[float]): The training loss over each epoch.
            val_loss (List[float]): The validation loss over each epoch.
        """
        self.y_pred = y_pred
        self.y_actual = y_actual
        self.label_encoder = label_encoder
        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
        self.train_loss = train_loss
        self.val_loss = val_loss

    def train_val_loss_accuracy_curves(self) -> None:
        """
        The training and validation loss curves as well as the training and
        validation accuracy curves. Used to evaluate for underfitting and
        overfitting of the model.
        """

        # Plots the training and validation accuracy over epochs.
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_accuracy, label='Training Accuracy')
        plt.plot(self.val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        # Plots the training and validation loss over epochs.
        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.tight_layout()
        plt.show()

    def confusion_matrix(self) -> None:
        """
        Creates a (C x C) confusion_matrix from sklearn. Then plots
        a heathmap of the confusion matrix using matplotlib and seaborn.
        """
        y_pred_numeric = np.argmax(self.y_pred, axis=1)
        cm = confusion_matrix(self.y_actual, y_pred_numeric)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_
                    )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def classification_metric(self):
        """
        General metrics including the accuracy, precision, recall, f_1 score
        and the ROC-curve.
        """
        y_pred_numeric = np.argmax(self.y_pred, axis=1)

        y_decoded_actual = self.label_encoder.inverse_transform(
            self.y_actual
        )
        y_decoded_pred = self.label_encoder.inverse_transform(y_pred_numeric)

        print(classification_report(y_decoded_actual, y_decoded_pred))

        plt.figure(figsize=(10, 7))
        for i in range(len(self.label_encoder.classes_)):
            fpr, tpr, _ = roc_curve(self.y_actual == i, self.y_pred[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label="Class" +
                     f"{self.label_encoder.classes_[i]}" +
                     f"(AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    def report_all_results(
            self,
    ):
        """
        These results include the training and validation loss curves as well
        as the training and validation accuracy curves. The confusion matrix
        for the machine learning model. Furthermore, general metrics including
        the accuracy, precision, recall, f_1 score and the ROC-curve.
        """

        # Only used for models that update their parameters overtime (gradient
        # descent), for the most part deep learning models.
        if not (self.train_accuracy is None or self.val_accuracy is None or
                self.train_loss is None or self.val_loss is None):
            self.train_val_loss_accuracy_curves()

        self.confusion_matrix()

        self.classification_metric()
