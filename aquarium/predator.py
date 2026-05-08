import numpy as np
from core.entity import Entity


class Predator(Entity):
    RADIUS = 16
    COLOR = (220, 50, 50)

    def __init__(self, position: np.ndarray, config: dict, entity_id: str = None):
        super().__init__(position, entity_id)
        pc = config.get("predator", {})
        self.speed: float = pc.get("speed", 1.8)
        self.detection_radius: float = pc.get("detection_radius", 180.0)
        self.kill_radius: float = pc.get("kill_radius", 14.0)
        self.max_kills: int = pc.get("max_kills", 6)
        self.lifetime: int = pc.get("lifetime", 1200)
        self.kills: int = 0
        self._age: int = 0
        self.velocity: np.ndarray = np.zeros(2, dtype=float)

    def get_radius(self) -> float:
        return self.RADIUS

    def update(self, environment) -> None:
        self._age += 1
        if self._age >= self.lifetime or self.kills >= self.max_kills:
            self.alive = False
            return

        # Import here to avoid circular import at module level
        from aquarium.fish_agent import FishAgent

        nearby = environment.get_nearby_entities(
            self.position, self.detection_radius, FishAgent
        )

        if nearby:
            target = min(nearby, key=lambda a: self.distance_to(a))
            direction = target.position - self.position
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                direction = direction / dist
            self.velocity = direction * self.speed
        else:
            # Drift slowly toward center if no target
            center = np.array([environment.width / 2, environment.height / 2])
            drift = center - self.position
            norm = np.linalg.norm(drift)
            if norm > 1e-6:
                self.velocity = (drift / norm) * (self.speed * 0.3)

        new_pos = self.position + self.velocity
        margin = self.RADIUS
        new_pos = np.clip(
            new_pos,
            [margin, margin],
            [environment.width - margin, environment.height - margin],
        )
        self.position = new_pos
