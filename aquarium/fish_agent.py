from collections import deque
from typing import List, Optional
import numpy as np

from core.agent import Agent
from brain.neural_brain import NeuralBrain
from genetics.genome import Genome


# ── Lineage colour registry ────────────────────────────────────────────────────
_LINEAGE_PALETTE = [
    (100, 180, 255), (100, 255, 140), (255, 220,  80), (255, 130,  80),
    (200, 100, 255), ( 80, 240, 220), (255, 100, 160), (200, 255, 100),
    (255, 200, 200), (140, 200, 255),
]
_lineage_color_map: dict = {}
_lineage_counter: list   = [0]


def _assign_lineage_color(lineage_id: str) -> tuple:
    if lineage_id not in _lineage_color_map:
        idx = _lineage_counter[0] % len(_LINEAGE_PALETTE)
        _lineage_color_map[lineage_id] = _LINEAGE_PALETTE[idx]
        _lineage_counter[0] += 1
    return _lineage_color_map[lineage_id]


def get_lineage_color_map() -> dict:
    return _lineage_color_map


# ── FishAgent ──────────────────────────────────────────────────────────────────

class FishAgent(Agent):
    RADIUS      = 7
    TRAIL_LEN   = 16   # max trail positions stored

    def __init__(
        self,
        position:    np.ndarray,
        brain:       NeuralBrain,
        genome:      Genome,
        config:      dict,
        entity_id:   str           = None,
        generation:  int           = 0,
        parent_ids:  List[str]     = None,
        lineage_id:  Optional[str] = None,
    ):
        super().__init__(position, entity_id, brain, genome, generation, parent_ids)
        ac = config.get("agent", {})
        sc = config.get("super_food", {})
        self.config = config

        self.energy:                  float = ac.get("initial_energy",          110.0)
        self.energy_decay:            float = ac.get("energy_decay",             0.03)
        self.reproduction_threshold:  float = ac.get("reproduction_threshold",  160.0)
        self.reproduction_cost:       float = ac.get("reproduction_cost",        60.0)
        self.reproduction_cooldown:   int   = ac.get("reproduction_cooldown",     100)
        self.max_speed:               float = ac.get("speed",                     3.0)
        self.detection_radius:        float = ac.get("detection_radius",        110.0)
        self.eat_radius:              float = ac.get("eat_radius",               18.0)
        self._steps_since_last_reproduction: int = self.reproduction_cooldown

        # Lineage & colour
        self.lineage_id: str   = lineage_id if lineage_id else self.id
        self.color             = _assign_lineage_color(self.lineage_id)

        # Predator bite state
        self.hits:          int = 0
        self.bite_cooldown: int = 0

        # Speed boost (from super food)
        self.speed_boost_remaining: int   = 0
        self.speed_boost_amount:    float = sc.get("speed_boost_amount", 1.8)

        # Voluntary sprint (neural output magnitude driven)
        self.sprint_threshold:    float = ac.get("sprint_threshold",    0.72)
        self.sprint_multiplier:   float = ac.get("sprint_multiplier",   1.7)
        self.sprint_energy_cost:  float = ac.get("sprint_energy_cost",  0.12)
        self.is_sprinting:        bool  = False

        # Critical energy penalty
        self.critical_energy:     float = ac.get("critical_energy",     35.0)
        self.low_energy_penalty:  float = ac.get("low_energy_penalty",  0.06)
        self.is_critical:         bool  = False   # visual flag

        # Danger signal
        self.is_signaling:          bool  = False
        self.signal_cooldown:       int   = 0
        self._signal_just_emitted:  bool  = False
        self.danger_signal_range:   float = ac.get("danger_signal_range", 90.0)

        # Motion trail  (each entry: (x, y, speed_ratio))
        self._trail: deque = deque(maxlen=self.TRAIL_LEN)

    def get_radius(self) -> float:
        return self.RADIUS

    # ── Perception (34-dim) ──────────────────────────────────────────────────

    def perceive(self, environment) -> np.ndarray:
        from aquarium.food import Food
        from aquarium.super_food import SuperFood
        from aquarium.predator import Predator

        def _encode(entities, n: int) -> List[float]:
            sorted_e = sorted(entities, key=lambda e: self.distance_to(e))[:n]
            vec = []
            for i in range(n):
                if i < len(sorted_e):
                    d = self.distance_to(sorted_e[i]) / (self.detection_radius + 1e-8)
                    a = self.angle_to(sorted_e[i])
                    vec.extend([min(d, 1.0), np.sin(a), np.cos(a)])
                else:
                    vec.extend([1.0, 0.0, 0.0])
            return vec

        def _encode_agents(nearby: list, n: int) -> List[float]:
            """Encode n nearest other agents: distance, sin, cos, can_reproduce (4 dims each)."""
            others   = [e for e in nearby if e.id != self.id]
            sorted_e = sorted(others, key=lambda e: self.distance_to(e))[:n]
            vec = []
            for i in range(n):
                if i < len(sorted_e):
                    d     = self.distance_to(sorted_e[i]) / (self.detection_radius + 1e-8)
                    a     = self.angle_to(sorted_e[i])
                    ready = 1.0 if sorted_e[i].can_reproduce() else 0.0
                    vec.extend([min(d, 1.0), np.sin(a), np.cos(a), ready])
                else:
                    vec.extend([1.0, 0.0, 0.0, 0.0])
            return vec

        # Treat super food as food for perception purposes (same channels)
        foods      = environment.get_nearby_entities(self.position, self.detection_radius, Food)
        sfoods     = environment.get_nearby_entities(self.position, self.detection_radius, SuperFood)
        all_food   = sorted(foods + sfoods, key=lambda e: self.distance_to(e))

        predators  = environment.get_nearby_entities(self.position, self.detection_radius, Predator)
        agents     = environment.get_nearby_entities(self.position, self.detection_radius, FishAgent)

        perception = []
        perception.extend(_encode(all_food,  3))               # 9  (3 × 3)
        perception.extend(_encode(predators, 2))               # 6  (2 × 3)
        perception.extend(_encode_agents(agents, 3))           # 12 (3 × 4: d,sin,cos,ready)
        perception.extend(environment.get_wall_distances(self.position).tolist())  # 4
        perception.extend([self.velocity[0] / self.max_speed,
                           self.velocity[1] / self.max_speed]) # 2
        perception.append(min(self.energy / 200.0, 1.0))       # 1
        # Total: 34

        return np.array(perception, dtype=float)

    # ── Action ───────────────────────────────────────────────────────────────

    def act(self, perception: np.ndarray) -> np.ndarray:
        raw_force = self.brain.think(perception)          # tanh → [-1,1]²
        magnitude = float(np.linalg.norm(raw_force))

        boost = self.speed_boost_amount if self.speed_boost_remaining > 0 else 1.0
        # Sprint only activates when NOT already speed-boosted by super food
        self.is_sprinting = (
            magnitude > self.sprint_threshold
            and self.speed_boost_remaining == 0
        )
        if self.is_sprinting:
            eff_speed = self.max_speed * self.sprint_multiplier
        else:
            eff_speed = self.max_speed * boost

        force = raw_force * eff_speed
        self.velocity = self.velocity * 0.6 + force * 0.4
        speed = np.linalg.norm(self.velocity)
        if speed > eff_speed:
            self.velocity = self.velocity / speed * eff_speed
        return self.velocity

    # ── Reproduction ─────────────────────────────────────────────────────────

    def can_reproduce(self) -> bool:
        return (
            self.energy >= self.reproduction_threshold
            and self._steps_since_last_reproduction >= self.reproduction_cooldown
        )

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, environment) -> None:
        if not self.alive:
            return

        self.age += 1
        self._steps_since_last_reproduction += 1

        # Counters
        if self.bite_cooldown > 0:
            self.bite_cooldown -= 1
        if self.speed_boost_remaining > 0:
            self.speed_boost_remaining -= 1

        # Danger signal logic
        self._signal_just_emitted = False
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            if self.signal_cooldown == 0:
                self.is_signaling = False
        else:
            from aquarium.predator import Predator
            preds = environment.get_nearby_entities(
                self.position, self.danger_signal_range, Predator
            )
            if preds and not self.is_signaling:
                self.is_signaling         = True
                self.signal_cooldown      = 90      # signal active for 90 steps
                self._signal_just_emitted = True

        # Base energy decay + sprint cost + critical penalty
        drain = self.energy_decay
        if self.is_sprinting:
            drain += self.sprint_energy_cost
        self.is_critical = self.energy < self.critical_energy
        if self.is_critical:
            drain += self.low_energy_penalty
        self.energy -= drain
        if self.energy <= 0:
            self.alive = False
            self.compute_fitness()
            return

        # Record position for trail BEFORE moving
        speed_ratio = float(np.linalg.norm(self.velocity)) / (self.max_speed + 1e-8)
        self._trail.append((self.position[0], self.position[1], speed_ratio))

        perception = self.perceive(environment)
        self.act(perception)

        # Soft wall repulsion — pushes velocity away before hitting the hard wall
        wz = self.RADIUS * 6  # repulsion zone width
        wx, wy = self.position[0], self.position[1]
        if wx < wz:
            self.velocity[0] += (wz - wx) / wz * self.max_speed * 0.5
        elif wx > environment.width - wz:
            self.velocity[0] -= (wx - (environment.width - wz)) / wz * self.max_speed * 0.5
        if wy < wz:
            self.velocity[1] += (wz - wy) / wz * self.max_speed * 0.5
        elif wy > environment.height - wz:
            self.velocity[1] -= (wy - (environment.height - wz)) / wz * self.max_speed * 0.5

        # Hard bounce: guarantee velocity points away from wall on contact
        margin  = self.RADIUS
        new_pos = self.position + self.velocity
        if new_pos[0] < margin:
            self.velocity[0] =  abs(self.velocity[0])
            new_pos[0] = margin
        elif new_pos[0] > environment.width - margin:
            self.velocity[0] = -abs(self.velocity[0])
            new_pos[0] = environment.width - margin
        if new_pos[1] < margin:
            self.velocity[1] =  abs(self.velocity[1])
            new_pos[1] = margin
        elif new_pos[1] > environment.height - margin:
            self.velocity[1] = -abs(self.velocity[1])
            new_pos[1] = environment.height - margin

        self.position = new_pos
