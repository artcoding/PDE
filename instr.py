from dataclasses import dataclass
import numpy as np

@dataclass
class Option:

    exercise: str = 'E'     # European
    type: str = 'C'         # Call
    ttm: float = 1.0        # 1 y to maturity
    strike: float = 1.0     # Strike (over spot)

    def payoff(self, spot: float) -> float:
        """
        Calculate the payoff of the option given the spot price.

        Parameters:
        spot : float
            The spot price of the underlying asset.

        Returns:
        float
            The payoff of the option.
        """
        if self.type == 'C':
            return np.maximum(spot - self.strike, 0.0)
        elif self.type == 'P':
            return np.maximum(self.strike - spot, 0.0)
        else:
            raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")
