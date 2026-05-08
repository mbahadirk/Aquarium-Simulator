from abc import ABC, abstractmethod
import numpy as np
import uuid


class Entity(ABC):
    def __init__(self, position: np.ndarray, entity_id: str = None):
        self.position = np.array(position, dtype=float)
        self.id = entity_id or str(uuid.uuid4())[:8]
        self.alive = True

    @abstractmethod
    def update(self, environment) -> None:
        pass

    @abstractmethod
    def get_radius(self) -> float:
        pass

    def distance_to(self, other: "Entity") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def distance_to_point(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - np.asarray(point)))

    def angle_to(self, other: "Entity") -> float:
        diff = other.position - self.position
        return float(np.arctan2(diff[1], diff[0]))

    def angle_to_point(self, point: np.ndarray) -> float:
        diff = np.asarray(point) - self.position
        return float(np.arctan2(diff[1], diff[0]))
