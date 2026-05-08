from abc import abstractmethod
from typing import List
import numpy as np

from core.entity import Entity


class Agent(Entity):
    def __init__(
        self,
        position: np.ndarray,
        entity_id: str = None,
        brain=None,
        genome=None,
        generation: int = 0,
        parent_ids: List[str] = None,
    ):
        super().__init__(position, entity_id)
        self.brain = brain
        self.genome = genome
        self.energy: float = 100.0
        self.age: int = 0
        self.generation: int = generation
        self.parent_ids: List[str] = parent_ids or []
        self.velocity: np.ndarray = np.zeros(2, dtype=float)
        self.children_count: int = 0
        self.total_food_eaten: int = 0
        self.fitness: float = 0.0

    @abstractmethod
    def perceive(self, environment) -> np.ndarray:
        pass

    @abstractmethod
    def act(self, perception: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def can_reproduce(self) -> bool:
        pass

    def compute_fitness(self) -> float:
        self.fitness = (
            self.age * 0.1
            + self.total_food_eaten * 5.0
            + self.children_count * 20.0
        )
        return self.fitness
