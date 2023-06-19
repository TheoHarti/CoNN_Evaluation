import numpy as np


def vertical_data(n_samples, n_classes):
    """vertical data set generator (for testing)"""
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype='uint8')
    for class_number in range(n_classes):
        ix = range(n_samples * class_number, n_samples * (class_number + 1))
        X[ix] = np.c_[np.random.randn(n_samples) * .1 + class_number / 3, np.random.randn(n_samples) * .1 + 0.5]
        y[ix] = class_number
    return X, y
