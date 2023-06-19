
import numpy as np


def klein_bottle_data(n_samples, noise, random_state):
    """Generate a Klein Bottle dataset with two classes."""
    # Set up the parameters for the Klein Bottle
    t = np.linspace(0, 2*np.pi, n_samples)
    a = 2
    b = 1
    x = (a + b*np.cos(t/2)*np.sin(t) - np.sin(t/2)*np.sin(2*t)) * np.cos(t)
    y = (a + b*np.cos(t/2)*np.sin(t) - np.sin(t/2)*np.sin(2*t)) * np.sin(t)
    z = b * np.cos(t/2) * np.sin(t) + np.sin(t/2) * np.sin(2*t)

    # Add noise to the data points
    x += noise * np.random.randn(n_samples)
    y += noise * np.random.randn(n_samples)
    z += noise * np.random.randn(n_samples)

    # Create binary labels for the data points
    # based on whether they are inside or outside the Klein Bottle
    label = np.zeros(n_samples)
    label[(x**2 + y**2 + z**2) < 1] = 1

    # Combine the data and labels into numpy arrays and return them
    X = np.column_stack((x, y))
    return X, label.astype(int)
