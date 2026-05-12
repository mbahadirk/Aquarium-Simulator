import numpy as np
from genetics.genome import Genome


def uniform_crossover(g1: Genome, g2: Genome) -> tuple:
    """Returns (child1, child2, r1, r2) where r = fraction of genes from g1."""
    assert g1.size == g2.size, "Genome sizes must match for crossover"
    mask = np.random.rand(g1.size) > 0.5
    c1 = np.where(mask, g1.weights, g2.weights)
    c2 = np.where(mask, g2.weights, g1.weights)
    r1 = float(mask.mean())       # fraction of c1 that came from g1
    r2 = float(1.0 - mask.mean()) # fraction of c2 that came from g1
    return Genome(c1), Genome(c2), r1, r2


def single_point_crossover(g1: Genome, g2: Genome) -> tuple:
    """Returns (child1, child2, r1, r2) where r = fraction of genes from g1."""
    assert g1.size == g2.size
    point = np.random.randint(1, g1.size)
    c1 = np.concatenate([g1.weights[:point], g2.weights[point:]])
    c2 = np.concatenate([g2.weights[:point], g1.weights[point:]])
    r1 = point / g1.size
    r2 = 1.0 - r1
    return Genome(c1), Genome(c2), r1, r2


def crossover(g1: Genome, g2: Genome, method: str = "uniform") -> tuple:
    if method == "uniform":
        return uniform_crossover(g1, g2)
    elif method == "single_point":
        return single_point_crossover(g1, g2)
    raise ValueError(f"Unknown crossover method: {method}")
