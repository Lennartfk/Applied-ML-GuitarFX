from GuitarFX.data.preprocessing import PreProcessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np


if __name__ == '__main__':
    """
    Code to run the baseline model. Will be moved to other part of code later.
    """
    pre_processing = PreProcessing(['datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon',
                                    'datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2'])
    
    X, y, feature_names, label_names = pre_processing.execute_mean_features()

    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df.to_csv('guitar_monophon_mean_features.csv', index=False)
    X_train_val, X_test, y_train_val, y_test, folds = pre_processing.data_splitting(X, y)

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    support_vector_machine = SVC()

    grid = GridSearchCV(support_vector_machine, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    grid.fit(X_train_val, y_train_val)

    print(grid.best_params_)
    print(grid.best_score_)

    support_vector_machine = SVC(kernel='rbf', C=1.0, gamma='scale')
    support_vector_machine.fit(X_train_val, y_train_val)
    y_pred = support_vector_machine.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Test set accuracy: {test_acc:.4f}")
