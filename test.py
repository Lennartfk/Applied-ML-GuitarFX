import numpy as np

data = np.load("data/cnn_scaled_onehot.npz")
y = data['y']
label_names = data['label_names']

# Count samples per class (since multilabel, count how many times each label appears)
class_counts = y.sum(axis=0)

for i, count in enumerate(class_counts):
    print(f"Class '{label_names[i]}': {int(count)} samples")
