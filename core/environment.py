from abc import ABC, abstractmethod
from typing import List, Type
import numpy as np

from core.entity import Entity


class Environment(ABC):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.entities: List[Entity] = []
        self.step_count = 0

    @abstractmethod
    def step(self) -> dict:
        pass

    @abstractmethod
    def spawn_food(self) -> None:
        pass

    @abstractmethod
    def spawn_predator(self) -> None:
        pass

    def add_entity(self, entity: Entity) -> None:
        self.entities.append(entity)

    def get_entities_of_type(self, entity_type: Type) -> List[Entity]:
        return [e for e in self.entities if isinstance(e, entity_type) and e.alive]

    def get_nearby_entities(
        self, position: np.ndarray, radius: float, entity_type: Type = None
    ) -> List[Entity]:
        result = []
        for e in self.entities:
            if not e.alive:
                continue
            if entity_type is not None and not isinstance(e, entity_type):
                continue
            if float(np.linalg.norm(e.position - position)) <= radius:
                result.append(e)
        return result

    def remove_dead_entities(self) -> List[Entity]:
        dead = [e for e in self.entities if not e.alive]
        self.entities = [e for e in self.entities if e.alive]
        return dead

    def get_wall_distances(self, position: np.ndarray) -> np.ndarray:
        return np.array([
            position[0] / self.width,
            (self.width - position[0]) / self.width,
            position[1] / self.height,
            (self.height - position[1]) / self.height,
        ], dtype=float)
