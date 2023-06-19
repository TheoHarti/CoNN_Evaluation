from dataclasses import dataclass


@dataclass
class PruningConfig:
    """data class for pruning specifications"""
    is_pruning_active: bool
    magnitude_threshold: float = 0.0