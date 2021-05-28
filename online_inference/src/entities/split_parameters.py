"""
dataclass for dataset split parameters
"""
from dataclasses import dataclass, field

@dataclass()
class SplittingParams:
    """
    Parameters
    """
    test_size: float = field(default=0.25)
    random_state: int = field(default=31)
