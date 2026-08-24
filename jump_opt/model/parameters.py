from dataclasses import dataclass


import numpy as np


@dataclass
class ModelParams:
    mp: float = 0.9
    mo: float = 0.6
    R: float = 0.12
    Lp: float = 0.05
    g: float = 9.81
    e: float = 0.5
    mu: float = 0.3


    @property
    def Io(self) -> float:
        return self.mo * self.R**2

    def as_array(self) -> np.ndarray:
        """
        Parameter ordering:
            [Io, mp, mo, Lp, g, R]
        """
        return np.array(
            [
                self.Io,
                self.mp,
                self.mo,
                self.Lp,
                self.g,
                self.R,
                self.e,
                self.mu,
            ],
            dtype=float,
        )

