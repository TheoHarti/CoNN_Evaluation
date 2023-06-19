import numpy as np


def corner_data():
    """corner data set generator (for testing)"""
    n = 15
    corner_data_pixel_format = np.concatenate((np.concatenate((np.zeros((n, n)), np.ones((n, n))), axis=1), np.ones((n, 2*n))), axis=0)
    corner_data_point_format = np.asarray(list(np.ndindex(corner_data_pixel_format.shape)))
    X = (corner_data_point_format - np.min(corner_data_point_format)) / np.ptp(corner_data_point_format)  # scale between 0 and 1
    y = corner_data_pixel_format.reshape((-1, 1)).flatten().astype(int)
    return X, y