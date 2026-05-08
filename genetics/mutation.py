import numpy as np
from genetics.genome import Genome


def gaussian_mutation(genome: Genome, rate: float = 0.08, strength: float = 0.12) -> Genome:
    """Apply per-gene Gaussian noise with probability `rate`."""
    weights = genome.weights.copy()
    mask = np.random.rand(len(weights)) < rate
    weights[mask] += np.random.randn(int(mask.sum())) * strength
    return Genome(weights)


def mutate(genome: Genome, config: dict) -> Genome:
    g = config.get("genetics", {})
    return gaussian_mutation(
        genome,
        rate=g.get("mutation_rate", 0.08),
        strength=g.get("mutation_strength", 0.12),
    )
