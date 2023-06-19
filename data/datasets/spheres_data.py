import numpy as np


def moons_data(n_samples, n_dimensions, n_clusters):
    """moons data set generator"""
    X = np.zeros((n_samples, n_dimensions))
    y = np.zeros(n_samples)

    for i in range(n_clusters):
        center = np.random.randn(n_dimensions)
        radius = np.random.rand() * 2 + 2  # Set the radius of the cluster

        indices = np.random.choice(n_samples, n_samples // n_clusters, replace=False)
        X[indices] = np.random.randn(n_samples // n_clusters, n_dimensions) * radius + center  # Add data points to the cluster with radius
        y[indices] = i

    return X, y
