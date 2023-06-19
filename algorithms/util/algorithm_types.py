from enum import Enum


class AlgorithmTypes(Enum):
    """The available algorithms that this evaluation system can use"""
    CasCor = 1
    CasCorUltra = 2
    Layerwise = 3
    UncertaintySplitting = 4
    ConstDeepNet = 5
    CCG_DLNN = 6
