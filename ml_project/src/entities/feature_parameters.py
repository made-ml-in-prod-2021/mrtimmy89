"""
dataclass for features
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    """
    Features
    """
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]
    use_log_trick: bool = field(default=False)