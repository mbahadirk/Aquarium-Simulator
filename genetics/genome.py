import numpy as np


class Genome:
    def __init__(self, weights: np.ndarray):
        self.weights = np.array(weights, dtype=float)

    @classmethod
    def random(cls, size: int) -> "Genome":
        return cls(np.random.randn(size) * 0.3)

    def clone(self) -> "Genome":
        return Genome(self.weights.copy())

    @property
    def size(self) -> int:
        return len(self.weights)

    def to_list(self) -> list:
        return self.weights.tolist()

    @classmethod
    def from_list(cls, data: list) -> "Genome":
        return cls(np.array(data, dtype=float))
