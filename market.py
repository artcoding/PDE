from dataclasses import dataclass

@dataclass
class Market:

    spot: float = 100.0 # spot
    r: float = 0.0      # interest rate (decimal)
    q: float = 0.0      # dividend yield
    vol: float = 1.0    # volatility
