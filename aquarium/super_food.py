import math
import numpy as np
from core.entity import Entity


class SuperFood(Entity):
    """Rare, high-value food that grants a temporary speed boost on consumption."""

    RADIUS = 9
    BASE_COLOR  = (255, 210, 60)   # gold
    GLOW_COLOR  = (255, 255, 160)  # bright centre

    def __init__(self, position: np.ndarray, config: dict, entity_id: str = None):
        super().__init__(position, entity_id)
        sc = config.get("super_food", {})
        self.energy_value: float      = sc.get("energy_value", 50.0)
        self.speed_boost_duration: int = sc.get("speed_boost_duration", 600)  # 10s @ 60fps
        self.speed_boost_amount: float = sc.get("speed_boost_amount", 1.8)
        self.lifetime: int             = sc.get("lifetime", 1200)
        self.drift_speed: float        = sc.get("drift_speed", 0.3)
        self._age: int = 0

    def get_radius(self) -> float:
        return self.RADIUS

    def update(self, environment) -> None:
        # No lifetime — super food stays until eaten
        if self.drift_speed > 0:
            dx = (np.random.rand() - 0.5) * 2 * self.drift_speed
            dy = (np.random.rand() - 0.5) * 2 * self.drift_speed
            self.position[0] = np.clip(self.position[0] + dx, 2, environment.width  - 2)
            self.position[1] = np.clip(self.position[1] + dy, 2, environment.height - 2)

    # ------------------------------------------------------------------
    # Animated colour (pulsing glow)
    # ------------------------------------------------------------------

    def pulse_radius(self, tick: int) -> int:
        """Outer glow radius oscillates for a pulsing effect."""
        return self.RADIUS + int(3 * abs(math.sin(tick / 20)))
