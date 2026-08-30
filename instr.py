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

    def can_exercise(self, t_tm: float) -> bool:
        """
        Check if the option can be exercised at time t.

        Parameters:
        t_tm : float
            Time to maturity (in years).

        Returns:
        bool
            True if the option can be exercised, False otherwise.
        """

        if self.exercise == 'E':
            return t_tm <= 0.0

        if self.exercise == 'A':
            return True
        
        raise ValueError("Invalid exercise type. Must be 'E' for European or 'A' for American.")
