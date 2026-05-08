from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn

from genetics.genome import Genome


class NeuralBrain(nn.Module):
    """Lightweight feedforward network that maps perception → movement force."""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        layers: List[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def think(self, perception: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.FloatTensor(perception).unsqueeze(0)
            return self.network(x).squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Genome serialisation
    # ------------------------------------------------------------------

    def get_flat_weights(self) -> np.ndarray:
        """Flatten all Linear weight + bias tensors into one 1-D array."""
        parts = []
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                parts.append(m.weight.data.numpy().flatten())
                parts.append(m.bias.data.numpy().flatten())
        return np.concatenate(parts)

    def load_genome(self, genome: Genome) -> None:
        idx = 0
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                rows, cols = m.weight.shape
                w_size = rows * cols
                m.weight.data = torch.FloatTensor(
                    genome.weights[idx: idx + w_size].reshape(rows, cols)
                )
                idx += w_size
                m.bias.data = torch.FloatTensor(genome.weights[idx: idx + rows])
                idx += rows

    def to_genome(self) -> Genome:
        return Genome(self.get_flat_weights())

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
    ) -> "NeuralBrain":
        brain = cls(input_size, hidden_sizes, output_size)
        brain.load_genome(genome)
        return brain

    @classmethod
    def from_config(cls, config: dict) -> "NeuralBrain":
        bc = config.get("brain", {})
        return cls(
            input_size=bc.get("input_size", 28),
            hidden_sizes=bc.get("hidden_sizes", [16, 8]),
            output_size=bc.get("output_size", 2),
        )

    @staticmethod
    def genome_size(input_size: int, hidden_sizes: List[int], output_size: int) -> int:
        """Total number of parameters (weights + biases) for this architecture."""
        sizes = [input_size] + hidden_sizes + [output_size]
        total = 0
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1] + sizes[i + 1]
        return total
