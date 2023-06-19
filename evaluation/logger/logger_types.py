from enum import Enum


class LoggerTypes(Enum):
    """The available loggers that this evaluation system can use"""
    Console = 1
    Tensorboard = 2
    Csv = 3