from dataclasses import dataclass


@dataclass
class InputOutputConfig:
    """data class for constructive network in and output specifications"""
    n_inputs: int
    n_outputs: int