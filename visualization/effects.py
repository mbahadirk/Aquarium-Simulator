"""Visual-only effects: wave rings and particle trails.

Nothing in this module modifies simulation state.
"""
from typing import List
import pygame


# ── Wave ring ─────────────────────────────────────────────────────────────────

class _Wave:
    MAX_RADIUS = 80
    RING_W     = 2
    COLOR      = (100, 220, 255)   # cyan-blue water ripple

    def __init__(self, x: int, y: int, max_age: int = 45):
        self.x = x
        self.y = y
        self.age = 0
        self.max_age = max_age

    def update(self) -> None:
        self.age += 1

    @property
    def alive(self) -> bool:
        return self.age < self.max_age

    def draw(self, surface: pygame.Surface) -> None:
        t = self.age / self.max_age          # 0 → 1
        radius = int(self.MAX_RADIUS * t)
        alpha  = int(220 * (1 - t) ** 1.5)
        if radius < 1:
            return
        r, g, b = self.COLOR
        pygame.draw.circle(surface, (r, g, b, alpha), (self.x, self.y), radius, self.RING_W)
        # Second, fainter outer ring
        if radius > 6:
            pygame.draw.circle(surface, (r, g, b, alpha // 3),
                               (self.x, self.y), radius + 4, 1)


# ── Effects manager ────────────────────────────────────────────────────────────

class EffectsManager:
    def __init__(self, width: int, height: int):
        self._waves: List[_Wave] = []
        self._surf = pygame.Surface((width, height), pygame.SRCALPHA)

    def add_wave(self, x: float, y: float) -> None:
        self._waves.append(_Wave(int(x), int(y)))

    def update(self) -> None:
        for w in self._waves:
            w.update()
        self._waves = [w for w in self._waves if w.alive]

    def draw(self, screen: pygame.Surface) -> None:
        self._surf.fill((0, 0, 0, 0))
        for w in self._waves:
            w.draw(self._surf)
        screen.blit(self._surf, (0, 0))
