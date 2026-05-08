import numpy as np
from core.entity import Entity


class Food(Entity):
    RADIUS = 5

    def __init__(self, position: np.ndarray, config: dict, entity_id: str = None):
        super().__init__(position, entity_id)
        fc = config.get("food", {})
        self.drift_speed: float = fc.get("drift_speed", 0.4)
        self.energy_value: float = fc.get("energy_value", 40.0)
        self.lifetime: int = fc.get("lifetime", 800)
        self._age: int = 0

    def get_radius(self) -> float:
        return self.RADIUS

    def update(self, environment) -> None:
        # No lifetime — food stays until eaten
        if self.drift_speed > 0:
            dx = (np.random.rand() - 0.5) * 2 * self.drift_speed
            dy = (np.random.rand() - 0.5) * 2 * self.drift_speed
            self.position[0] = np.clip(self.position[0] + dx, 2, environment.width - 2)
            self.position[1] = np.clip(self.position[1] + dy, 2, environment.height - 2)
