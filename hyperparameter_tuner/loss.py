from abc import ABC
from typing import Dict, Any

import pandas as pd


class Loss(ABC):
    """Interface for loss functions."""

    def __call__(self, values: pd.Series, params: Dict[str, Any]) -> float:
        """
        Calculate the loss based on the provided DataFrame row.

        Args:
            row: pd.Series - The specified values from the value dataframe.
        Returns:
            float - The calculated loss.
        """
        pass
