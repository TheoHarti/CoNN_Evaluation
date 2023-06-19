import numpy as np


def spiral_data(n_points, n_classes):
    """spirals data set generator"""
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype='uint8')
    for class_number in range(n_classes):
        ix = range(n_points * class_number, n_points * (class_number + 1))
        r = np.linspace(0.0, 1, n_points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    p = np.random.permutation(len(y))
    return X[p], y[p]
