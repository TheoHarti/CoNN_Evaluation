from enum import Enum


class DatasetTypes(Enum):
    """The available datasets that this evaluation system can use"""
    Corner = 0
    Vertical = 0.1
    Spirals2 = 1.1
    Spirals3 = 1.2
    Spirals4 = 1.3
    Curves = 2
    Spheres = 3
    Compound = 4
    Wine = 10
    Digits = 11
    BreastCancer = 12
    MNIST = 13
    OlivettiFaces = 14