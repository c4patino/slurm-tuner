from abc import ABC, abstractmethod

import pandas as pd


class Loss(ABC):
    """Interface for loss functions."""

    @abstractmethod
    def calculate(self, row: pd.Series) -> float:
        """
        Calculate the loss based on the provided DataFrame row.

        Args:
            row: pd.Series - A row from the DataFrame.
        Returns:
            float - The calculated loss.
        """
        pass
