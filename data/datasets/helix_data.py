
import numpy as np


def helix_data(n_samples, noise):
    """helix data set generator"""
    t_max = 30 * np.pi
    radius = 1
    z = np.linspace(0, 1, n_samples)
    theta = np.linspace(0, t_max, n_samples)

    X = np.zeros((n_samples, 3))
    X[:, 0] = radius * np.cos(theta)
    X[:, 1] = radius * np.sin(theta)
    X[:, 2] = z + noise * np.random.randn(n_samples)
    y = np.zeros(n_samples)
    y[:n_samples // 2] = 1
    return X, y
