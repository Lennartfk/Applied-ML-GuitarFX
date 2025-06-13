from typing import List, Optional, Union

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    hamming_loss,
)


class ModelMetrics:
    """
    Reports results for a multi-label classification machine learning model.
    """
    def __init__(
        self,
        y_pred: Union[np.ndarray, List[float]],
        y_actual: Union[np.ndarray, List[float]],
        y_pred_probs: Optional[Union[np.ndarray, List[float]]] = None,
        label_names: Optional[List[Union[str, List[str]]]] = None,
        threshold: float = 0.5,
        train_accuracy: Optional[List[float]] = None,
        val_accuracy: Optional[List[float]] = None,
        train_loss: Optional[List[float]] = None,
        val_loss: Optional[List[float]] = None,
        val_auc: Optional[List[float]] = None,
        train_auc: Optional[List[float]] = None
    ) -> None:
        """
        Initializes the attributes of the ModelMetrics class.

        Args:
            y_pred (Union[np.ndarray, List[float]]): Predicted class labels
                from the model.
            y_actual (Union[np.ndarray, List[float]]): Ground truth class
                labels.
            y_pred_probs (Optional[Union[np.ndarray, List[float]]], optional):
                Predicted class probabilities (useful for metrics like AUC).
            label_names (Optional[List[Union[str, List[str]]]], optional):
                Names of the labels/classes.
            threshold (float, optional): Threshold for converting
                probabilities to class predictions (used in binary
                classification). Defaults to 0.5.
            train_accuracy (Optional[List[float]], optional): List of training
                accuracy values (e.g., across epochs).
            val_accuracy (Optional[List[float]], optional):
                List of validation accuracy values (e.g., across epochs).
            train_loss (Optional[List[float]], optional):
                List of training loss values.
            val_loss (Optional[List[float]], optional): List of validation
                loss values.
            val_auc (Optional[List[float]], optional): List of validation AUC
                scores.
            train_auc (Optional[List[float]], optional): List of training AUC
                scores.
        """
        self.y_pred_probs = np.array(y_pred)
        self.y_actual = np.array(y_actual)
        self.threshold = threshold
        self.y_pred = (self.y_pred_probs >= threshold).astype(int)

        num_classes = self.y_actual.shape[1]
        if y_pred_probs is not None:
            self.y_pred_probs = np.array(y_pred_probs)
            self.y_pred = (self.y_pred_probs >= threshold).astype(int)
        else:
            self.y_pred = np.array(y_pred)
            self.y_pred_probs = self.y_pred

        if label_names is None:
            self.label_names = [f"Class {i}" for i in range(num_classes)]
        elif isinstance(label_names, list) and all(isinstance(label, str)
                                                   for label in label_names):
            if len(label_names) != num_classes:
                print(f"[Warning] Provided {len(label_names)} label names, " +
                      f"but expected {num_classes}. Using default " +
                      "labels instead.")
                self.label_names = [f"Class {i}" for i in range(num_classes)]
            else:
                self.label_names = label_names
        else:
            print("[Warning] Invalid label_names format." +
                  "Using default labels.")
            self.label_names = [f"Class {i}" for i in range(num_classes)]

        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.val_auc = val_auc
        self.train_auc = train_auc

    def train_val_loss_auc_curves(self) -> None:
        """
        Reports the training and validation loss curves and the training and
        validation ROC curves over epochs using matplotlib.
        """
        if (
            self.train_loss is None or
            self.val_loss is None or
            self.train_auc is None or
            self.val_auc is None
        ):
            print("Training/Validation loss/AUC data not provided.")
            return

        plt.figure(figsize=(14, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")

        # AUC plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_auc, label="Training AUC")
        plt.plot(self.val_auc, label="Validation AUC")
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.legend()
        plt.title("Training and Validation AUC")

        plt.tight_layout()
        plt.show()

    def _confusion_matrix_per_class(self) -> np.ndarray:
        """
        Creates a confusion matrix for each class.

        Returns:
            np.ndarray: dictionary with class name keys and confusion matrix
            values (2x2) for each class, since multi-label confusion matrix is
            per class.
        """
        cms = {}
        n_classes = self.y_actual.shape[1]
        for i, class_name in enumerate(self.label_names):
            if i >= n_classes:
                print(f"[Warning] Skipping label '{class_name}' - index" +
                      f"{i} exceeds number of classes ({n_classes})")
                continue
            cms[class_name] = confusion_matrix(
                self.y_actual[:, i], self.y_pred[:, i], labels=[0, 1]
            )
        return cms

    def plot_confusion_matrices(self) -> None:
        """
        Plot confusion matrices for each class in a grid.
        """
        cms = self._confusion_matrix_per_class()
        n_classes = len(self.label_names)
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols

        plt.figure(figsize=(5 * n_cols, 4 * n_rows))

        for idx, (class_name, cm) in enumerate(cms.items()):
            plt.subplot(n_rows, n_cols, idx + 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"])
            plt.title(f"Confusion Matrix: {class_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

        plt.tight_layout()
        plt.show()

    def plot_per_class_accuracy(self) -> None:
        """
        Per-class accuracy = TP+TN / total for each class
        """
        cms = self._confusion_matrix_per_class()
        accuracies = []
        for cm in cms.values():
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            accuracies.append(accuracy)

        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_acc = np.array(accuracies)[sorted_indices]
        sorted_classes = np.array(self.label_names)[sorted_indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_acc, y=sorted_classes, palette="viridis")
        plt.xlim(0, 1)
        plt.xlabel("Accuracy")
        plt.ylabel("Class")
        plt.title("Per-Class Accuracy")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def classification_metrics_report(self) -> None:
        """
        Prints precision, recall, f1-score per class and averages.
        Also prints hamming loss and exact match accuracy.
        """
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_actual, self.y_pred, zero_division=0
        )
        print("Per-class Precision, Recall, F1-score and Support:\n")
        for i, class_name in enumerate(self.label_names):
            print(
                f"{class_name}: Precision={precision[i]:.3f},"
                f"  Recall={recall[i]:.3f}, "
                f"F1={f1[i]:.3f}, Support={support[i]}"
            )

        # Micro average
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            self.y_actual, self.y_pred, average="micro", zero_division=0
        )
        print(
            f"\nMicro-avg Precision={p_micro:.3f}, Recall={r_micro:.3f}, "
            f"F1={f1_micro:.3f}"
        )

        # Macro average
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            self.y_actual, self.y_pred, average="macro", zero_division=0
        )
        print(
            f"Macro-avg Precision={p_macro:.3f}, Recall={r_macro:.3f}, "
            f"F1={f1_macro:.3f}"
        )

        # Exact match accuracy (subset accuracy)
        exact_match_acc = np.mean(np.all(self.y_pred == self.y_actual, axis=1))
        print(f"\nExact match accuracy: {exact_match_acc:.4f}")

        # Hamming loss
        hl = hamming_loss(self.y_actual, self.y_pred)
        print(f"Hamming loss: {hl:.4f}")

    def plot_roc_curves(self) -> None:
        """
        Plots ROC curves for each class with AUC.
        """
        plt.figure(figsize=(10, 7))
        for i, class_name in enumerate(self.label_names):
            fpr, tpr, _ = roc_curve(
                self.y_actual[:, i],
                self.y_pred_probs[:, i]
            )
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.grid(True)
        plt.show()

    def classification_metrics_report_single_vs_multi(self) -> None:
        """
        Prints metrics separately for single-label and multi-label samples.
        """
        # Identify single-label and multi-label indices
        single_label_indices = [i for i, label_vec in enumerate(self.y_actual)
                                if label_vec.sum() == 1]
        multi_label_indices = [i for i, label_vec in enumerate(self.y_actual)
                               if label_vec.sum() > 1]

        if single_label_indices:
            print("\n=== Metrics for SINGLE-LABEL samples ===")
            y_actual_single = self.y_actual[single_label_indices]
            y_pred_single = self.y_pred[single_label_indices]

            precision, recall, f1, support = precision_recall_fscore_support(
                y_actual_single, y_pred_single, zero_division=0
            )
            for i, class_name in enumerate(self.label_names):
                print(
                    f"{class_name}: Precision={precision[i]:.3f}, "
                    f"Recall={recall[i]:.3f}, "
                    f"F1={f1[i]:.3f}, Support={support[i]}"
                )
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                y_actual_single,
                y_pred_single,
                average="micro",
                zero_division=0
            )
            print(
                f"Micro-avg Precision={p_micro:.3f}, Recall={r_micro:.3f}, "
                f"F1={f1_micro:.3f}"
            )
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                y_actual_single,
                y_pred_single,
                average="macro",
                zero_division=0
            )
            print(
                f"Macro-avg Precision={p_macro:.3f}, Recall={r_macro:.3f}, "
                f"F1={f1_macro:.3f}"
            )
            exact_match_acc = np.mean(np.all(y_pred_single == y_actual_single,
                                             axis=1))
            print(f"Exact match accuracy: {exact_match_acc:.4f}")
            hl = hamming_loss(y_actual_single, y_pred_single)
            print(f"Hamming loss: {hl:.4f}")

        if multi_label_indices:
            print("\n=== Metrics for MULTI-LABEL samples ===")
            y_actual_multi = self.y_actual[multi_label_indices]
            y_pred_multi = self.y_pred[multi_label_indices]

            precision, recall, f1, support = precision_recall_fscore_support(
                y_actual_multi, y_pred_multi, zero_division=0
            )
            for i, class_name in enumerate(self.label_names):
                print(
                    f"{class_name}: Precision={precision[i]:.3f}, "
                    f"Recall={recall[i]:.3f}, "
                    f"F1={f1[i]:.3f}, Support={support[i]}"
                )
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                y_actual_multi, y_pred_multi, average="micro", zero_division=0
            )
            print(
                f"Micro-avg Precision={p_micro:.3f}, Recall={r_micro:.3f}, "
                f"F1={f1_micro:.3f}"
            )
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                y_actual_multi, y_pred_multi, average="macro", zero_division=0
            )
            print(
                f"Macro-avg Precision={p_macro:.3f}, Recall={r_macro:.3f}, "
                f"F1={f1_macro:.3f}"
            )
            exact_match_acc = np.mean(np.all(y_pred_multi == y_actual_multi,
                                             axis=1))
            print(f"Exact match accuracy: {exact_match_acc:.4f}")
            hl = hamming_loss(y_actual_multi, y_pred_multi)
            print(f"Hamming loss: {hl:.4f}")

    def report_all_results(self) -> None:
        """
        Generate all reports and plots.
        """
        if (
            self.train_loss is not None and
            self.val_loss is not None and
            self.train_auc is not None and
            self.val_auc is not None
        ):
            self.train_val_loss_auc_curves()

        self.plot_confusion_matrices()
        self.plot_per_class_accuracy()
        self.classification_metrics_report()
        self.plot_roc_curves()
