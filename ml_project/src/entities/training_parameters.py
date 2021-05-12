"""
dataclass for training parameters
"""
from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    """
    Parameters
    """
    model_type: str = field(default="LogisticRegression")
    random_state: int = field(default=42)
